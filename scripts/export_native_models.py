"""Prepare fixed weights for standalone C++/CUDA inference using real inputs."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
from torch import nn

from geosunpre.artifacts import file_metadata, write_json
from geosunpre.features import FRONTEND_CONFIG, LoMaFrontend, read_feature_cache
from geosunpre.matching import forward_matches
from geosunpre.pipeline.task import configure_offline


class Detector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        result = self.model(image, 12288)
        return result['keypoints'].float(), result['keypoint_probs'].float()


class Descriptor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, keypoints):
        # Keep the pinned descriptor's sampling and mixed precision path.
        dense = self.model(image)
        return torch.nn.functional.grid_sample(dense.float(), keypoints[:, None],
            mode='bilinear', align_corners=False)[:, :, 0].mT.float()


class Matcher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ka, kb, da, db):
        matches, confidence = forward_matches(self.model(ka, kb, da, db)['scores'], .1)
        return matches, confidence.float()


def fixed_descriptor_positions(model, image, keypoints):
    # The production descriptor always receives 784 x 784 images. Compute the
    # unchanged DINO position interpolation once, outside the tracing path.
    vit = model.encoder.frozen_dinov2.dinov2_vitl14
    with torch.inference_mode():
        expected = Descriptor(model)(image, keypoints)
        patches = vit.patch_embed(image.to(vit.cls_token.dtype))
        tokens = torch.cat((vit.cls_token.expand(image.shape[0], -1, -1), patches), dim=1)
        positions = vit.interpolate_pos_encoding(tokens, 784, 784)
        vit.register_buffer('_native_positions', positions)
        def interpolate(self, x, width, height):
            return self._native_positions.to(x.dtype)
        vit.interpolate_pos_encoding = MethodType(interpolate, vit)
        actual = Descriptor(model)(image, keypoints)
        if not torch.equal(expected, actual):
            raise RuntimeError('Fixed DINO position interpolation changed the descriptor')


class Segmentation(nn.Module):
    def __init__(self, model, sky, person):
        super().__init__()
        self.model = model
        self.sky, self.person = sky, person

    def forward(self, image):
        probability = self.model(image, return_dict=False)[0].softmax(1)[:, [self.sky, self.person]]
        return torch.nn.functional.interpolate(probability, (512, 512),
            mode='bilinear', align_corners=False)


def _tuple(value):
    return value if isinstance(value, tuple) else (value,)


def convert(label, module, inputs, root):
    """Check each deployed graph against its real-image eager result."""
    started = time.perf_counter()
    print(f'Export {label}: tracing real inputs', flush=True)
    module.eval()
    with torch.inference_mode():
        expected = _tuple(module(*inputs))
        path = root / f'{label}.pt'
        if not path.is_file():
            traced = torch.jit.trace(module, inputs, check_trace=False)
            traced = torch.jit.freeze(traced, optimize_numerics=False)
            traced.save(str(path))
        loaded = torch.jit.load(str(path), map_location='cuda').eval()
        actual = _tuple(loaded(*inputs))
        differences = []
        for index, (before, after) in enumerate(zip(expected, actual, strict=True)):
            difference = float((before.float() - after.float()).abs().max())
            equal = torch.equal(before, after) if not before.is_floating_point() else torch.allclose(
                before, after, atol=1e-4 if label == 'segmentation' else 1e-5, rtol=1e-4)
            differences.append({'max_absolute_error': difference, 'passed': bool(equal)})
            if label == 'segmentation':
                threshold = before.new_tensor([.90, .55]).reshape(1, 2, 1, 1)
                changed = int(((before > threshold) != (after > threshold)).sum())
                differences[-1]['threshold_pixels_changed'] = changed
                equal = bool(equal) and changed == 0
                differences[-1]['passed'] = bool(equal)
            if not equal:
                raise RuntimeError(f'{label}: converted output {index} differs: {difference}')
        names = []
        for index, tensor in enumerate(inputs):
            name = f'{label}_input_{index}.npy'
            np.save(root / 'examples' / name, tensor.float().cpu().numpy())
            names.append(name)
        for index, tensor in enumerate(actual):
            np.save(root / 'examples' / f'{label}_output_{index}.npy', tensor.cpu().numpy())
    print(f'Export {label}: verified, {time.perf_counter() - started:.1f}s', flush=True)
    return {'file': file_metadata(path), 'inputs': names,
            'input_shapes': [list(t.shape) for t in inputs],
            'outputs': len(actual), 'real_input_comparison': differences,
            'seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-cache', type=Path, default=Path('.cache/huggingface'))
    parser.add_argument('--resume', action='store_true', help='Recheck the graphs of an unfinished conversion')
    args = parser.parse_args()
    configure_offline()
    os.environ['TORCH_HOME'] = str(Path('.cache/torch').resolve())
    if not torch.cuda.is_available():
        raise RuntimeError('Model conversion requires the CUDA production device')
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f'Choose a fresh model output: {root}')
    pending = root.with_name(root.name + '.pending')
    if pending.exists() and not args.resume:
        raise FileExistsError(f'Unfinished model conversion: {pending}')
    (pending / 'examples').mkdir(parents=True, exist_ok=args.resume)
    scene = json.loads((args.work / 'sfm/manifest.json').read_text(encoding='utf-8'))
    frontend = LoMaFrontend(args.work, scene, FRONTEND_CONFIG)
    model = frontend.load_model()
    torch._C._jit_set_autocast_mode(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    record_a, record_b = scene['images'][:2]
    detector, descriptor = (value.cuda() for value in frontend.prepare_image(record_a))
    features = [read_feature_cache(args.work / 'features' / f'{r["name"]}.npz', r['name'])
                for r in (record_a, record_b)]
    ka, kb = (torch.from_numpy(f['normalized_keypoints'])[None].cuda() for f in features)
    da, db = (torch.from_numpy(f['descriptors'])[None].cuda() for f in features)
    reports = {}
    reports['detector'] = convert('detector', Detector(model._detector), (detector,), pending)
    fixed_descriptor_positions(model._descriptor, descriptor, ka)
    reports['descriptor'] = convert('descriptor', Descriptor(model._descriptor), (descriptor, ka), pending)
    reports['matcher'] = convert('matcher', Matcher(model), (ka, kb, da, db), pending)
    # Matching and descriptor graphs must accept actual subsets of retained points.
    with torch.inference_mode():
        smaller = (ka[:, :1024], kb[:, :1536], da[:, :1024], db[:, :1536])
        loaded = torch.jit.load(str(pending / 'matcher.pt'), map_location='cuda')
        reference = Matcher(model)(*smaller)
        deployed = loaded(*smaller)
        if not torch.equal(reference[0], deployed[0]) or not torch.allclose(reference[1], deployed[1], atol=1e-5, rtol=1e-4):
            raise RuntimeError('Matching graph does not preserve variable feature counts')
        reports['matcher']['variable_feature_counts_verified'] = [1024, 1536]
    from geosunpre.masks import _load_segmentation, virtual_views
    from geosunpre.camera import Camera
    import cv2
    from PIL import Image
    processor, segmentation = _load_segmentation(args.model_cache)
    sky = next(int(k) for k, v in segmentation.config.id2label.items() if v == 'sky')
    person = next(int(k) for k, v in segmentation.config.id2label.items() if v == 'person')
    cam = Camera.from_dict(scene['cameras'][record_a['camera']])
    with Image.open(record_a['source']) as image:
        rgb = np.asarray(image.convert('RGB'))
    source = next(virtual_views(cam))[2]
    view = cv2.remap(rgb, source[..., 0], source[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
    pixels = processor(images=Image.fromarray(view), return_tensors='pt')['pixel_values'].cuda()
    reports['segmentation'] = convert('segmentation', Segmentation(segmentation, sky, person), (pixels,), pending)
    write_json(pending / 'manifest.json', {'version': 1, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'frontend': asdict(FRONTEND_CONFIG),
        'attention': 'efficient', 'segmentation': {'sky_id': sky, 'person_id': person,
            'mean': processor.image_mean, 'std': processor.image_std},
        'models': reports, 'runtime': 'C++ LibTorch/CUDA; no Python execution',
        'source_scope': 'Pinned production weights and two real selected exposures.'})
    pending.rename(root)
    print(f'Native models: {root}', flush=True)


if __name__ == '__main__':
    main()
