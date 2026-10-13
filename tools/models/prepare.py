"""Download, export, build and verify resources for native C++ inference."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch
import onnx
import tensorrt as trt

from resources import CACHE, loma_source, segmentation_source, sha256, torch_weights
from networks import Detector, Descriptor, Matcher, Segmentation, fix_descriptor_positions

TRT_VERSION = '10.13.3.9'
MAX_FEATURES = 3072
LOGGER = trt.Logger(trt.Logger.WARNING)
CONTRACTS = {
    'detector': (['image'], ['logits']),
    'descriptor': (['image', 'keypoints'], ['descriptors']),
    'matcher': (['ka', 'kb', 'da', 'db', 'mask_a', 'mask_b'], ['matches', 'confidence']),
    'segmentation': (['image'], ['probabilities']),
}


def write_json(path, value):
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def image_tensor(path, width, height):
    with Image.open(path) as image:
        pixels = np.array(image.convert('RGB').resize((width, height), Image.Resampling.BICUBIC))
    return torch.from_numpy(pixels).permute(2, 0, 1)[None].cuda().float() / 255.


def keypoints(logits, count=MAX_FEATURES):
    pooled = torch.nn.functional.max_pool2d(logits, 3, 1, 1)
    scores = torch.where(logits == pooled, logits, -torch.inf).flatten()
    indices = scores.topk(count).indices
    height, width = logits.shape[-2:]
    xy = torch.stack((indices % width + .5, indices // width + .5), dim=1)
    return (2 * xy / xy.new_tensor([width, height]) - 1).float()[None]


def padded(ka, kb, da, db, count=MAX_FEATURES):
    na, nb = ka.shape[1], kb.shape[1]
    values = []
    for value in (ka, kb, da, db):
        output = torch.zeros((1, count, value.shape[-1]), device='cuda')
        output[:, :value.shape[1]] = value
        values.append(output)
    ma, mb = (torch.zeros((1, count), device='cuda') for _ in range(2))
    ma[:, :na], mb[:, :nb] = 1, 1
    return (*values, ma, mb)


def load_loma(source):
    sys.path.insert(0, str(source / 'src'))
    weights = torch_weights()
    from loma import LoMa, LoMaB
    import loma.descriptor.dedode as dedode
    import loma.detector.dad as dad
    # Export in float32; TensorRT selects half precision for dense models.
    dedode.amp_dtype = dad.amp_dtype = torch.float32
    original = torch.hub.load_state_dict_from_url

    def local_weights(url, **kwargs):
        return torch.load(weights[url.rsplit('/', 1)[-1]], map_location='cpu', weights_only=True)

    torch.hub.load_state_dict_from_url = local_weights
    try:
        model = LoMa(LoMaB(compile=False, mp=False)).eval().float().cuda()
    finally:
        torch.hub.load_state_dict_from_url = original
    for module in model.modules():
        if hasattr(module, 'amp'):
            module.amp = False
        if hasattr(module, 'amp_dtype'):
            module.amp_dtype = torch.float32
    fix_descriptor_positions(model._descriptor)
    return model


def expected_shapes(width, height):
    return {
        'detector': [[1, 3, height, width]],
        'descriptor': [[1, 3, 784, 784], [1, MAX_FEATURES, 2]],
        'matcher': [[1, MAX_FEATURES, 2], [1, MAX_FEATURES, 2],
                    [1, MAX_FEATURES, 256], [1, MAX_FEATURES, 256],
                    [1, MAX_FEATURES], [1, MAX_FEATURES]],
        'segmentation': [[1, 3, 512, 512]],
    }


def check_onnx(path, name, shapes):
    graph = onnx.load(str(path), load_external_data=False)
    if any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer):
        raise RuntimeError(f'Requires a single-file ONNX with embedded weights: {path}')
    inputs, outputs = CONTRACTS[name]
    if [v.name for v in graph.graph.input] != inputs or [v.name for v in graph.graph.output] != outputs:
        raise RuntimeError(f'ONNX names differ from the native contract: {path}')
    for value, expected in zip(graph.graph.input, shapes):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f'ONNX input must be float32: {path}: {value.name}')
        dims = value.type.tensor_type.shape.dim
        if len(dims) != len(expected):
            raise RuntimeError(f'ONNX input rank differs: {path}: {value.name}')
        for axis, (dim, size) in enumerate(zip(dims, expected)):
            if name == 'descriptor' and value.name == 'keypoints' and axis == 1:
                if not dim.dim_param:
                    raise RuntimeError(f'Descriptor keypoints must have a dynamic count: {path}')
            elif dim.dim_value != size:
                raise RuntimeError(f'ONNX input size differs: {path}: {value.name}')
    onnx.checker.check_model(str(path))


def parse_network(path):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    return builder, network, parser


def export_graph(name, module, inputs, directory, shapes, references):
    path = directory / f'{name}.onnx'
    if path.is_file():
        check_onnx(path, name, shapes)
        print(f'Reuse ONNX: {name}', flush=True)
        return
    print(f'Export ONNX: {name}', flush=True)
    temporary = path.with_name(path.name + '.writing')
    dynamic = {'keypoints': {1: 'N'}, 'descriptors': {1: 'N'}} if name == 'descriptor' else None
    with torch.inference_mode():
        expected = module(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        reference = {f'input{i}': value.cpu().numpy() for i, value in enumerate(inputs)}
        reference.update({f'output{i}': value.cpu().numpy() for i, value in enumerate(expected)})
        np.savez(references / f'{name}.npz', **reference)
        torch.onnx.export(module.eval(), inputs, str(temporary), opset_version=17,
                          input_names=CONTRACTS[name][0], output_names=CONTRACTS[name][1],
                          dynamic_axes=dynamic, dynamo=False)
    check_onnx(temporary, name, shapes)
    # Reject incompatible exports before spending time tuning any engines.
    parsed = parse_network(temporary)
    del parsed
    temporary.replace(path)


def export_models(directory, shapes, source, images, references):
    required = [name for name in CONTRACTS if not (directory / f'{name}.onnx').is_file()]
    if any(name != 'segmentation' for name in required):
        if source is None:
            source = loma_source()
        model = load_loma(source)
        detector_image = image_tensor(images[0], shapes['detector'][0][3], shapes['detector'][0][2])
        descriptor_image = image_tensor(images[0], 784, 784)
        with torch.inference_mode():
            ka = keypoints(Detector(model._detector)(detector_image))
            da = Descriptor(model._descriptor)(descriptor_image, ka)
            # Prove the masked adapter preserves upstream unpadded inference.
            from loma.loma import filter_matches
            small = (ka[:, :127], ka[:, :193], da[:, :127], da[:, :193])
            expected, _, scores, _ = filter_matches(model(*small)['scores'], .1)
            actual, confidence = Matcher(model)(*padded(*small, count=193))
            if not torch.equal(expected, actual[:, :127].long()) or not torch.allclose(scores, confidence[:, :127], atol=1e-5, rtol=1e-4):
                raise RuntimeError('Masked matcher adapter differs from upstream unpadded inference')
        for name, module, inputs in (
            ('detector', Detector(model._detector), (detector_image,)),
            ('descriptor', Descriptor(model._descriptor), (descriptor_image, ka[:, :127])),
            ('matcher', Matcher(model), padded(ka, ka, da, da)),
        ):
            if name in required:
                export_graph(name, module, inputs, directory, shapes[name], references)
        del model, module, inputs, detector_image, descriptor_image, ka, da
        gc.collect()
        torch.cuda.empty_cache()
    if 'segmentation' in required:
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(segmentation_source(), local_files_only=True).eval().cuda()
        mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
        pixels = (image_tensor(images[0], 512, 512) - mean) / std
        export_graph('segmentation', Segmentation(model), (pixels,), directory, shapes['segmentation'], references)
        del model, pixels
        gc.collect()
        torch.cuda.empty_cache()


def build_engine(path, name, shapes):
    if path.is_file():
        print(f'Reuse engine: {name}', flush=True)
        return
    print(f'Build TensorRT engine: {name} (this can take several minutes)', flush=True)
    onnx_path = path.parent.parent / 'onnx' / f'{name}.onnx'
    builder, network, parser = parse_network(onnx_path)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if name != 'matcher':
        config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    if name == 'descriptor':
        profile = builder.create_optimization_profile()
        profile.set_shape('keypoints', (1, 32, 2), (1, 1024, 2), (1, MAX_FEATURES, 2))
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT build failed: {name}')
    temporary = path.with_name(path.name + '.writing')
    temporary.write_bytes(bytes(serialized))
    temporary.replace(path)


class Engine:
    def __init__(self, root, name, shapes):
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine((root / 'engines' / f'{name}.engine').read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Engine incompatible with GPU/runtime: {name}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f'Cannot create execution context: {name}')
        self.inputs, self.outputs = CONTRACTS[name]
        actual_inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                         if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        actual_outputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                          if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        if set(actual_inputs) != set(self.inputs) or set(actual_outputs) != set(self.outputs):
            raise RuntimeError(f'Engine tensor names differ: {name}')
        for label, shape in zip(self.inputs, shapes):
            actual = tuple(self.engine.get_tensor_shape(label))
            expected = (1, -1, 2) if name == 'descriptor' and label == 'keypoints' else tuple(shape)
            if actual != expected or self.engine.get_tensor_dtype(label) != trt.float32:
                raise RuntimeError(f'Engine input contract differs: {name}: {label}: {actual}')

    def __call__(self, *inputs):
        inputs = [value.contiguous() for value in inputs]
        for label, value in zip(self.inputs, inputs):
            if not self.context.set_input_shape(label, tuple(value.shape)):
                raise RuntimeError(f'Cannot set engine shape: {label}: {tuple(value.shape)}')
            self.context.set_tensor_address(label, value.data_ptr())
        outputs = []
        dtypes = {trt.float32: torch.float32, trt.int32: torch.int32, trt.int64: torch.int64}
        for label in self.outputs:
            dtype = self.engine.get_tensor_dtype(label)
            if dtype not in dtypes:
                raise RuntimeError(f'Unsupported engine output type: {label}: {dtype}')
            output = torch.empty(tuple(self.context.get_tensor_shape(label)), device='cuda', dtype=dtypes[dtype])
            self.context.set_tensor_address(label, output.data_ptr())
            outputs.append(output)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT inference failed')
        torch.cuda.synchronize()
        if any(not torch.isfinite(value).all().item() for value in outputs):
            raise RuntimeError('Nonfinite TensorRT output')
        return tuple(outputs)


def compare_reference(root, name, shapes, references):
    path = references / f'{name}.npz'
    if not path.is_file():
        return  # --onnx-from verifies existing exported graphs by their runtime behavior.
    engine = Engine(root, name, shapes[name])
    with np.load(path) as example, torch.inference_mode():
        inputs = [torch.from_numpy(example[f'input{i}']).cuda() for i in range(len(CONTRACTS[name][0]))]
        outputs = engine(*inputs)
        for i, actual in enumerate(outputs):
            expected = torch.from_numpy(example[f'output{i}']).cuda()
            if expected.shape != actual.shape:
                raise RuntimeError(f'Engine output differs from the export reference: {name}')
            if not expected.is_floating_point():
                passed = torch.equal(expected, actual)
                error = int((expected != actual).sum())
            elif name == 'segmentation':
                error = float((expected - actual).abs().max())
                passed = error < .03
            else:
                error = float((expected - actual).norm() / expected.norm().clamp_min(1e-8))
                passed = error < (.005 if name == 'matcher' else .02)
            if not passed:
                raise RuntimeError(f'TensorRT output differs from PyTorch: {name}/{i}: error={error}')
            print(f'PyTorch/TensorRT comparison: {name}/{i}: error={error:.6g}', flush=True)


def installed_runtime():
    import tensorrt_libs
    return Path(tensorrt_libs.__file__).parent / 'nvinfer_10.dll'


def verify(root, shapes, images):
    if sha256(root / 'runtime' / 'nvinfer_10.dll') != sha256(installed_runtime()):
        raise RuntimeError('Packaged runtime differs from the pinned TensorRT DLL')
    print('Verify all four engines with real images, dynamic counts and padding...', flush=True)
    with torch.inference_mode():
        detector = Engine(root, 'detector', shapes['detector'])
        points = []
        for path in images:
            logits, = detector(image_tensor(path, shapes['detector'][0][3], shapes['detector'][0][2]))
            if tuple(logits.shape) != (1, 1, shapes['detector'][0][2], shapes['detector'][0][3]):
                raise RuntimeError('Detector output shape differs')
            points.append(keypoints(logits))
        del detector
        descriptor = Engine(root, 'descriptor', shapes['descriptor'])
        descriptions = []
        for path, k in zip(images, points):
            output, = descriptor(image_tensor(path, 784, 784), k)
            if tuple(output.shape) != (1, MAX_FEATURES, 256) or output.norm().item() == 0:
                raise RuntimeError('Descriptor output shape/value differs')
            descriptions.append(output)
        for count in (32, 127):
            smaller, = descriptor(image_tensor(images[0], 784, 784), points[0][:, :count])
            if tuple(smaller.shape) != (1, count, 256) or not torch.allclose(smaller, descriptions[0][:, :count], atol=1e-4, rtol=1e-4):
                raise RuntimeError(f'Descriptor dynamic count changed its values: {count}')
        del descriptor
        matcher = Engine(root, 'matcher', shapes['matcher'])
        matched = 0
        for na, nb in ((32, 64), (127, 193), (MAX_FEATURES, MAX_FEATURES)):
            inputs = padded(points[0][:, :na], points[1][:, :nb],
                            descriptions[0][:, :na], descriptions[1][:, :nb])
            matches, confidence = matcher(*inputs)
            if tuple(matches.shape) != (1, MAX_FEATURES) or matches.dtype != torch.int32 or tuple(confidence.shape) != (1, MAX_FEATURES):
                raise RuntimeError('Matcher output contract differs')
            if ((matches[:, :na] < -1) | (matches[:, :na] >= nb)).any() or (matches[:, na:] != -1).any():
                raise RuntimeError('Matcher produced an invalid/padded match')
            if (confidence < 0).any() or (confidence > 1.00001).any():
                raise RuntimeError('Matcher confidence is outside [0,1]')
            # Changing invalid values must not affect attention or double softmax.
            changed = [value.clone() for value in inputs]
            for index, count in enumerate((na, nb, na, nb)):
                changed[index][:, count:] = .75
            again, score = matcher(*changed)
            if not torch.equal(matches[:, :na], again[:, :na]) or not torch.allclose(confidence[:, :na], score[:, :na], atol=1e-5, rtol=1e-4):
                raise RuntimeError(f'Matcher padding changed valid features: {na}/{nb}')
            matched = int((matches >= 0).sum())
        identity, _ = matcher(*padded(points[0], points[0], descriptions[0], descriptions[0]))
        valid = identity[0] >= 0
        if not valid.any() or not torch.equal(identity[0, valid].long(), torch.arange(MAX_FEATURES, device='cuda')[valid]):
            raise RuntimeError('Matcher failed real-image identity matching')
        del matcher
        segmentation = Engine(root, 'segmentation', shapes['segmentation'])
        mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
        for path in images:
            probabilities, = segmentation((image_tensor(path, 512, 512) - mean) / std)
            if tuple(probabilities.shape) != (1, 2, 512, 512) or (probabilities < 0).any() or (probabilities > 1.00001).any() or (probabilities.sum(1) > 1.00001).any():
                raise RuntimeError('Segmentation output contract differs')
    print(f'Verified: four engines; descriptor N=32/127/3072; matcher unequal counts and padding; real pair matches={matched}', flush=True)


def verify_package(root, settings, shapes, images):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or any(manifest.get(key) != settings[key] for key in ('gpu', 'tensorrt', 'detector_size')):
        raise RuntimeError('Existing model GPU/version/resolution differs. Choose a new -Models directory.')
    for name, (inputs, outputs) in CONTRACTS.items():
        spec = manifest['models'][name]
        if spec['inputs'] != inputs or spec['outputs'] != outputs or spec['shapes'] != shapes[name]:
            raise RuntimeError(f'Existing model manifest contract differs: {name}')
        if sha256(root / 'engines' / f'{name}.engine') != spec['sha256']:
            raise RuntimeError(f'Existing engine checksum differs: {name}. Choose a new -Models directory.')
    if not (root / 'runtime' / 'nvinfer_10.dll').is_file():
        raise RuntimeError('Existing model runtime DLL is missing')
    verify(root, shapes, images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--image-width', type=int, default=3000)
    parser.add_argument('--image-height', type=int, default=4000)
    parser.add_argument('--onnx-from', type=Path)
    parser.add_argument('--images', type=Path, nargs=2)
    args = parser.parse_args()
    if trt.__version__ != TRT_VERSION:
        raise RuntimeError(f'Requires TensorRT {TRT_VERSION}, found {trt.__version__}')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for model preparation')
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    scale = 1536 / max(args.image_width, args.image_height)
    width, height = (int(value * scale) // 8 * 8 for value in (args.image_width, args.image_height))
    if min(width, height) < 8:
        raise RuntimeError('Image aspect ratio produces an empty detector input')
    shapes = expected_shapes(width, height)
    root = args.models.resolve()
    recipe = ''.join(sha256(Path(__file__).with_name(name)) for name in ('prepare.py', 'networks.py', 'resources.py', 'requirements.txt'))
    settings = {'gpu': torch.cuda.get_device_name(0), 'tensorrt': TRT_VERSION,
                'detector_size': [width, height],
                'recipe': hashlib.sha256(recipe.encode('ascii')).hexdigest()}
    if args.onnx_from:
        origin = args.onnx_from.resolve()
        if (origin / 'onnx').is_dir():
            origin = origin / 'onnx'
        settings['onnx_from'] = {name: sha256(origin / f'{name}.onnx') for name in CONTRACTS}
    source = None if args.images else loma_source()
    images = args.images or [source / 'assets' / f'0015_{letter}.jpg' for letter in ('A', 'B')]
    images = [path.resolve() for path in images]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (root / 'manifest.json').is_file():
        verify_package(root, settings, shapes, images)
        print(f'Reuse complete model package: {root}', flush=True)
        return
    if root.exists():
        raise RuntimeError('Output already exists without a manifest. Choose a new -Models directory.')
    pending = root.with_name(root.name + '.pending')
    if (pending / 'manifest.json').is_file():
        verify_package(pending, settings, shapes, images)
        print(f'Reuse verified pending model package: {pending}', flush=True)
        return
    settings_path = pending / 'preparation.json'
    if pending.is_dir() and any(pending.iterdir()) and not settings_path.is_file():
        raise RuntimeError(f'Existing pending directory has no preparation record: {pending}. Choose a new -Models directory.')
    pending.mkdir(parents=True, exist_ok=True)
    if settings_path.is_file() and json.loads(settings_path.read_text(encoding='utf-8')) != settings:
        raise RuntimeError(f'Unfinished conversion has different settings: {pending}. Choose a new -Models directory.')
    write_json(settings_path, settings)
    references = CACHE / 'references' / sha256(settings_path)
    references.mkdir(parents=True, exist_ok=True)
    for directory in ('onnx', 'engines', 'runtime'):
        (pending / directory).mkdir(exist_ok=True)
    if args.onnx_from:
        origin = args.onnx_from.resolve()
        if (origin / 'onnx').is_dir():
            origin = origin / 'onnx'
        for name in CONTRACTS:
            candidate = origin / f'{name}.onnx'
            check_onnx(candidate, name, shapes[name])
            destination = pending / 'onnx' / candidate.name
            if not destination.exists():
                shutil.copy2(candidate, destination)
            elif sha256(destination) != settings['onnx_from'][name]:
                raise RuntimeError(f'Unfinished ONNX differs from the requested source: {destination}')
    export_models(pending / 'onnx', shapes, source, images, references)
    for name in CONTRACTS:
        check_onnx(pending / 'onnx' / f'{name}.onnx', name, shapes[name])
        build_engine(pending / 'engines' / f'{name}.engine', name, shapes[name])
        gc.collect()
        torch.cuda.empty_cache()
    runtime = installed_runtime()
    shutil.copy2(runtime, pending / 'runtime' / runtime.name)
    for package, name in (('tensorrt-cu12-libs', 'LICENSE.tensorrt.txt'),
                          ('nvidia-cuda-runtime-cu12', 'LICENSE.cuda.txt')):
        distribution = importlib.metadata.distribution(package)
        license_file = next(file for file in distribution.files if file.name.lower() in ('license.txt', 'license'))
        shutil.copy2(distribution.locate_file(license_file), pending / 'runtime' / name)
    for name in CONTRACTS:
        compare_reference(pending, name, shapes, references)
    verify(pending, shapes, images)
    manifest = {'version': 1, 'tensorrt': TRT_VERSION, 'gpu': settings['gpu'],
                'detector_size': [width, height], 'descriptor_size': [784, 784],
                'max_features': MAX_FEATURES, 'descriptor_dimension': 256,
                'segmentation_mean': [.485, .456, .406], 'segmentation_std': [.229, .224, .225],
                'models': {name: {'inputs': names[0], 'outputs': names[1], 'shapes': shapes[name],
                                 'sha256': sha256(pending / 'engines' / f'{name}.engine')}
                           for name, names in CONTRACTS.items()}}
    write_json(pending / 'manifest.json', manifest)
    # Publish in PowerShell after this process releases any Windows file handles.
    print(f'Model package verified: {pending}', flush=True)


if __name__ == '__main__':
    main()
