"""Download, export, build and verify resources for native C++ inference."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch
import onnx
import tensorrt as trt

from resources import (CACHE, GEOMETRY_REVISION, LOMA_COMMIT, MOGE_COMMIT, SEGMENTATION_REVISION, geometry_weights,
                       loma_source, moge_source, segmentation_source, sha256, torch_weights)
from networks import Detector, Descriptor, Geometry, Matcher, Segmentation, fix_descriptor_positions, fix_geometry_positions

TRT_VERSION = '10.13.3.9'
MAX_FEATURES = 3072
DETECTOR_PROFILE = {'min': [16, 16], 'opt': [1152, 1536], 'max': [1536, 1536]}
DETECTOR_CHECK_SIZES = ((1152, 1536), (1536, 1152), (1536, 864), (1000, 1536),
                        (1536, 1536), (16, 1536), (1536, 16))
# MoGe-2 runs on square pinhole faces: 46x46 ViT tokens, inside the suggested 1200-2500.
GEOMETRY_SIZE = 644
GEOMETRY_TOKENS = 2116
LOGGER = trt.Logger(trt.Logger.WARNING)
CONTRACTS = {
    'detector': (['image'], ['logits']),
    'descriptor': (['image', 'keypoints'], ['descriptors']),
    'matcher': (['ka', 'kb', 'da', 'db', 'mask_a', 'mask_b'], ['matches', 'confidence']),
    'segmentation': (['image'], ['probabilities']),
    'geometry': (['image'], ['points', 'normal', 'mask', 'metric_scale']),
}
# Version 1 ONNX bundles predate the geometry model; it is exported from the pinned weights.
BUNDLE_MODELS = {1: [name for name in CONTRACTS if name != 'geometry'], 2: list(CONTRACTS)}
BUNDLE = 'onnx-bundle.json'
# Hardware compatibility trades some tactics for one engine set on every Ampere+ GPU.
HARDWARE_LEVELS = {'none': None, 'ampere_plus': (8, 0)}
REFERENCES = [f'{name}.npz' for name in CONTRACTS] + [
    f'detector_{width}x{height}.npz' for width, height in DETECTOR_CHECK_SIZES] + ['geometry_B.npz']


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


def load_geometry():
    os.environ['XFORMERS_DISABLED'] = '1'
    sys.path.insert(0, str(moge_source()))
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(str(geometry_weights())).eval().float().cuda()
    generator = torch.Generator('cuda').manual_seed(0)
    image = torch.rand((1, 3, GEOMETRY_SIZE, GEOMETRY_SIZE), device='cuda', generator=generator)
    with torch.inference_mode():
        upstream = Geometry(model, GEOMETRY_TOKENS)(image)
        fix_geometry_positions(model, GEOMETRY_SIZE)
        exported = Geometry(model, GEOMETRY_TOKENS)(image)
    # At the fixed size the ONNX-mode resize is an identity; the frozen positions must match upstream.
    for before, after in zip(upstream, exported):
        if not torch.allclose(before, after, atol=1e-5, rtol=1e-4):
            raise RuntimeError('Export-mode geometry model differs from upstream inference')
    return Geometry(model, GEOMETRY_TOKENS)


def detector_size(width, height):
    size = [value * 1536 // max(width, height) // 8 * 8 for value in (width, height)]
    if min(size) < 16:
        raise RuntimeError('Image aspect ratio produces a detector side below 16 pixels')
    return size


def expected_shapes():
    return {
        'detector': [[1, 3, -1, -1]],
        'descriptor': [[1, 3, 784, 784], [1, MAX_FEATURES, 2]],
        'matcher': [[1, MAX_FEATURES, 2], [1, MAX_FEATURES, 2],
                    [1, MAX_FEATURES, 256], [1, MAX_FEATURES, 256],
                    [1, MAX_FEATURES], [1, MAX_FEATURES]],
        'segmentation': [[1, 3, 512, 512]],
        'geometry': [[1, 3, GEOMETRY_SIZE, GEOMETRY_SIZE]],
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
            if size == -1 or (name == 'descriptor' and value.name == 'keypoints' and axis == 1):
                if not dim.dim_param:
                    raise RuntimeError(f'ONNX input axis must be dynamic: {path}: {value.name}/{axis}')
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
    if name == 'detector':
        dynamic = {'image': {2: 'H', 3: 'W'}, 'logits': {2: 'H', 3: 'W'}}
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
    if any(name in ('detector', 'descriptor', 'matcher') for name in required):
        if source is None:
            source = loma_source()
        model = load_loma(source)
        detector_image = image_tensor(images[0], *DETECTOR_PROFILE['opt'])
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
                if name == 'detector':
                    with torch.inference_mode():
                        for width, height in DETECTOR_CHECK_SIZES:
                            pixels = image_tensor(images[0], width, height)
                            logits = module(pixels)
                            np.savez(references / f'detector_{width}x{height}.npz',
                                     input0=pixels.cpu().numpy(), output0=logits.cpu().numpy())
                    del pixels, logits
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
    if 'geometry' in required:
        module = load_geometry()
        pixels = image_tensor(images[0], GEOMETRY_SIZE, GEOMETRY_SIZE)
        export_graph('geometry', module, (pixels,), directory, shapes['geometry'], references)
        with torch.inference_mode():
            pixels = image_tensor(images[1], GEOMETRY_SIZE, GEOMETRY_SIZE)
            np.savez(references / 'geometry_B.npz', input0=pixels.cpu().numpy(),
                     **{f'output{i}': value.cpu().numpy() for i, value in enumerate(module(pixels))})
        del module, pixels
        gc.collect()
        torch.cuda.empty_cache()


def build_engine(path, name, shapes, matcher_precision='fp16', hardware='none'):
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
    if name == 'matcher' and matcher_precision == 'tf32':
        config.set_flag(trt.BuilderFlag.TF32)
    elif name == 'matcher' and matcher_precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        scoring = False
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            # Softmax cannot recover precision already lost in its score inputs.
            # Keep final projections, similarity and match selection in FP32.
            scoring = scoring or layer.name == '/final_proj/MatMul'
            sensitive = layer.type in (trt.LayerType.SOFTMAX, trt.LayerType.REDUCE, trt.LayerType.NORMALIZATION)
            score_operation = scoring and layer.type in (
                trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE,
                trt.LayerType.SELECT, trt.LayerType.TOPK)
            if sensitive or score_operation:
                floating = [i for i in range(layer.num_outputs)
                            if layer.get_output(i).dtype == trt.float32]
                if not floating:
                    continue
                layer.precision = trt.float32
                for output in floating:
                    layer.set_output_type(output, trt.float32)
        if not scoring:
            raise RuntimeError('FP16 matcher ONNX is missing the final scoring projection')
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if hardware == 'ampere_plus':
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    if name == 'detector':
        profile = builder.create_optimization_profile()
        profile.set_shape('image', *[(1, 3, height, width)
                                    for width, height in DETECTOR_PROFILE.values()])
        config.add_optimization_profile(profile)
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
        if name == 'detector':
            profile = [tuple(dims) for dims in self.engine.get_tensor_profile_shape('image', 0)]
            expected = [(1, 3, height, width) for width, height in DETECTOR_PROFILE.values()]
            if profile != expected:
                raise RuntimeError(f'Detector optimization profile differs: {profile}')

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


def geometry_errors(expected, actual):
    """Depth, normal, mask and scale differences where the PyTorch mask is valid."""
    valid = expected[2][0] > .5
    if valid.sum() < 1000:
        raise RuntimeError('Geometry reference has too few valid pixels')
    depth = (actual[0][0, ..., 2].clamp_min(1e-6).log() - expected[0][0, ..., 2].log()).abs()[valid]
    cosine = (actual[1][0] * expected[1][0]).sum(-1).clamp(-1, 1)[valid]
    return {'log_depth_p99': float(depth.quantile(.99)),
            'normal_degrees_p99': float(torch.rad2deg(cosine.acos()).quantile(.99)),
            'mask_p99': float((actual[2] - expected[2]).abs().flatten().quantile(.99)),
            'scale': float(((actual[3] - expected[3]).abs() / expected[3]).max())}


def compare_reference(root, name, shapes, references):
    if name in ('detector', 'geometry'):
        paths = sorted(references.glob(f'{name}*.npz'))
    else:
        paths = [references / f'{name}.npz']
    paths = [path for path in paths if path.is_file()]
    if not paths:
        return  # --onnx-from verifies existing exported graphs by their runtime behavior.
    engine = Engine(root, name, shapes[name])
    for path in paths:
        with np.load(path) as example, torch.inference_mode():
            inputs = [torch.from_numpy(example[f'input{i}']).cuda() for i in range(len(CONTRACTS[name][0]))]
            outputs = engine(*inputs)
            if name == 'geometry':
                expected = [torch.from_numpy(example[f'output{i}']).cuda() for i in range(len(outputs))]
                if [value.shape for value in expected] != [value.shape for value in outputs]:
                    raise RuntimeError(f'Engine output differs from the export reference: {path.stem}')
                errors = geometry_errors(expected, outputs)
                limits = {'log_depth_p99': .01, 'normal_degrees_p99': 1., 'mask_p99': .05, 'scale': .005}
                if any(errors[key] >= limit for key, limit in limits.items()):
                    raise RuntimeError(f'TensorRT geometry differs from PyTorch: {path.stem}: {errors}')
                print(f'PyTorch/TensorRT comparison: {path.stem}: ' +
                      ', '.join(f'{key}={value:.6g}' for key, value in errors.items()), flush=True)
                continue
            for i, actual in enumerate(outputs):
                expected = torch.from_numpy(example[f'output{i}']).cuda()
                if expected.shape != actual.shape:
                    raise RuntimeError(f'Engine output differs from the export reference: {path.stem}')
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
                    raise RuntimeError(f'TensorRT output differs from PyTorch: {path.stem}/{i}: error={error}')
                print(f'PyTorch/TensorRT comparison: {path.stem}/{i}: error={error:.6g}', flush=True)


def installed_runtime():
    import tensorrt_libs
    return Path(tensorrt_libs.__file__).parent / 'nvinfer_10.dll'


def verify(root, shapes, images):
    if sha256(root / 'runtime' / 'nvinfer_10.dll') != sha256(installed_runtime()):
        raise RuntimeError('Packaged runtime differs from the pinned TensorRT DLL')
    print('Verify all five engines with real images, dynamic counts and padding...', flush=True)
    with torch.inference_mode():
        detector = Engine(root, 'detector', shapes['detector'])
        points = []
        for path in images:
            with Image.open(path) as image:
                width, height = detector_size(*image.size)
            logits, = detector(image_tensor(path, width, height))
            if tuple(logits.shape) != (1, 1, height, width):
                raise RuntimeError('Detector output shape differs')
            points.append(keypoints(logits))
        for width, height in DETECTOR_CHECK_SIZES:
            logits, = detector(image_tensor(images[0], width, height))
            if tuple(logits.shape) != (1, 1, height, width):
                raise RuntimeError(f'Dynamic detector output shape differs: {width}x{height}')
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
        del segmentation
        geometry = Engine(root, 'geometry', shapes['geometry'])
        size = GEOMETRY_SIZE
        for path in images:
            points, normal, mask, scale = geometry(image_tensor(path, size, size))
            if (tuple(points.shape) != (1, size, size, 3) or tuple(normal.shape) != (1, size, size, 3) or
                    tuple(mask.shape) != (1, size, size) or tuple(scale.shape) != (1,)):
                raise RuntimeError('Geometry output contract differs')
            valid = mask[0] > .5
            if valid.float().mean() < .5 or (points[0, ..., 2][valid] <= 0).any() or scale.item() <= 0:
                raise RuntimeError('Geometry output is not a valid point map')
            if ((normal.norm(dim=-1) - 1).abs() > 1e-3).any() or (mask < 0).any() or (mask > 1).any():
                raise RuntimeError('Geometry normals or mask are out of range')
    print(f'Verified: five engines; dynamic detector shapes; descriptor N=32/127/3072; matcher unequal counts and padding; real pair matches={matched}; geometry point maps', flush=True)


def matcher_precisions(manifest):
    precision = manifest.get('matcher_precision', 'fp32')
    return precision, manifest.get('matcher_score_precision', precision)


def hardware_level(manifest):
    return manifest.get('hardware_compatibility', 'none')


def same_target(manifest, settings):
    """Engines from the same TensorRT version run on the build GPU model or any GPU of their level."""
    if manifest.get('tensorrt') != settings['tensorrt'] or hardware_level(manifest) != settings['hardware_compatibility']:
        return False
    return settings['hardware_compatibility'] != 'none' or manifest.get('gpu') == settings['gpu']


def verify_package(root, settings, shapes, images):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 3 or not same_target(manifest, settings) or manifest.get('detector_profile') != settings['detector_profile']:
        raise RuntimeError('Existing model GPU/hardware level/version/profile differs. Choose a new -Models directory; use -OnnxFrom to upgrade a static package.')
    if matcher_precisions(manifest) != matcher_precisions(settings):
        raise RuntimeError('Existing matcher calculation/scoring precision differs. Choose a new -Models directory; use -OnnxFrom to reuse exported graphs.')
    for name, (inputs, outputs) in CONTRACTS.items():
        spec = manifest['models'][name]
        if spec['inputs'] != inputs or spec['outputs'] != outputs or spec['shapes'] != shapes[name]:
            raise RuntimeError(f'Existing model manifest contract differs: {name}')
        if sha256(root / 'engines' / f'{name}.engine') != spec['sha256']:
            raise RuntimeError(f'Existing engine checksum differs: {name}. Choose a new -Models directory.')
    if not (root / 'runtime' / 'nvinfer_10.dll').is_file():
        raise RuntimeError('Existing model runtime DLL is missing')
    verify(root, shapes, images)


def read_bundle(root):
    bundle = json.loads((root / BUNDLE).read_text(encoding='utf-8'))
    if bundle.get('version') not in BUNDLE_MODELS or bundle.get('detector_profile') != DETECTOR_PROFILE or bundle.get('max_features') != MAX_FEATURES:
        raise RuntimeError(f'ONNX bundle version/profile differs from this preparation tool: {root}')
    if bundle['version'] == 2 and (bundle.get('geometry_size') != GEOMETRY_SIZE or bundle.get('geometry_tokens') != GEOMETRY_TOKENS):
        raise RuntimeError(f'ONNX bundle geometry input differs from this preparation tool: {root}')
    models = BUNDLE_MODELS[bundle['version']]
    references = [name for name in REFERENCES if name.split('_')[0].split('.')[0] in models]
    required = [f'onnx/{name}.onnx' for name in models] + [f'references/{name}' for name in references] + bundle['images']
    files = bundle['files']
    if len(bundle['images']) != 2 or any(name not in files for name in required):
        raise RuntimeError(f'ONNX bundle is incomplete: {root}')
    for name, digest in files.items():
        if sha256(root / name) != digest:
            raise RuntimeError(f'ONNX bundle checksum differs: {root / name}')
    return bundle


def export_bundle(root, shapes, source, images):
    """Export the GPU-independent ONNX graphs with PyTorch references and validation images."""
    if root.exists():
        raise RuntimeError(f'ONNX bundle output already exists: {root}. Choose a new -OnnxBundle directory.')
    pending = root.with_name(root.name + '.pending')
    if (pending / BUNDLE).is_file():
        read_bundle(pending)
        print(f'Reuse verified pending ONNX bundle: {pending}', flush=True)
        return
    recipe = ''.join(sha256(Path(__file__).with_name(name)) for name in ('prepare.py', 'networks.py', 'resources.py', 'requirements.txt'))
    settings = {'recipe': hashlib.sha256(recipe.encode('ascii')).hexdigest(), 'images': [sha256(path) for path in images]}
    settings_path = pending / 'preparation.json'
    if pending.is_dir() and any(pending.iterdir()) and (not settings_path.is_file() or json.loads(settings_path.read_text(encoding='utf-8')) != settings):
        raise RuntimeError(f'Unfinished ONNX bundle has different settings: {pending}. Choose a new -OnnxBundle directory.')
    for directory in ('onnx', 'references', 'images'):
        (pending / directory).mkdir(parents=True, exist_ok=True)
    write_json(settings_path, settings)
    export_models(pending / 'onnx', shapes, source, images, pending / 'references')
    missing = [name for name in REFERENCES if not (pending / 'references' / name).is_file()]
    if missing:
        raise RuntimeError(f'Unfinished ONNX bundle lacks PyTorch references {missing}. Choose a new -OnnxBundle directory.')
    for name in CONTRACTS:
        check_onnx(pending / 'onnx' / f'{name}.onnx', name, shapes[name])
    names = []
    for letter, path in zip('AB', images):
        names.append(f'images/{letter}{path.suffix.lower()}')
        shutil.copy2(path, pending / names[-1])
    files = [f'onnx/{name}.onnx' for name in CONTRACTS] + [f'references/{name}' for name in REFERENCES] + names
    write_json(pending / BUNDLE, {
        'version': 2, 'recipe': settings['recipe'], 'loma_commit': LOMA_COMMIT,
        'segmentation_revision': SEGMENTATION_REVISION, 'moge_commit': MOGE_COMMIT,
        'geometry_revision': GEOMETRY_REVISION, 'detector_profile': DETECTOR_PROFILE,
        'max_features': MAX_FEATURES, 'geometry_size': GEOMETRY_SIZE,
        'geometry_tokens': GEOMETRY_TOKENS, 'images': names,
        'files': {name: sha256(pending / name) for name in files}})
    read_bundle(pending)
    print(f'ONNX bundle verified: {pending}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--bundle', type=Path, help='export a GPU-independent ONNX bundle instead of engines')
    parser.add_argument('--matcher-precision', choices=('fp32', 'tf32', 'fp16'), default='fp16',
                        help='matcher engine precision (default: %(default)s)')
    parser.add_argument('--hardware-compatibility', choices=tuple(HARDWARE_LEVELS), default='none',
                        help='TensorRT engine hardware compatibility (default: %(default)s)')
    parser.add_argument('--onnx-from', type=Path)
    parser.add_argument('--images', type=Path, nargs=2)
    args = parser.parse_args()
    if bool(args.models) == bool(args.bundle) or (args.bundle and args.onnx_from):
        raise RuntimeError('Use either --models (optionally with --onnx-from) or --bundle')
    if trt.__version__ != TRT_VERSION:
        raise RuntimeError(f'Requires TensorRT {TRT_VERSION}, found {trt.__version__}')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for model preparation')
    torch.cuda.set_device(0)
    minimum = HARDWARE_LEVELS[args.hardware_compatibility]
    if minimum and torch.cuda.get_device_capability(0) < minimum:
        raise RuntimeError(f'{args.hardware_compatibility} engines require a GPU with compute capability >= {minimum[0]}.{minimum[1]}')
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    shapes = expected_shapes()
    recipe = ''.join(sha256(Path(__file__).with_name(name)) for name in ('prepare.py', 'networks.py', 'resources.py', 'requirements.txt'))
    settings = {'gpu': torch.cuda.get_device_name(0), 'tensorrt': TRT_VERSION,
                'detector_profile': DETECTOR_PROFILE,
                'hardware_compatibility': args.hardware_compatibility,
                'matcher_precision': args.matcher_precision,
                'matcher_score_precision': 'fp32' if args.matcher_precision == 'fp16' else args.matcher_precision,
                'recipe': hashlib.sha256(recipe.encode('ascii')).hexdigest()}
    reuse_manifest = bundle = None
    if args.onnx_from:
        origin = args.onnx_from.resolve()
        if (origin / 'onnx').is_dir():
            origin = origin / 'onnx'
        if (origin.parent / BUNDLE).is_file():
            bundle = read_bundle(origin.parent)
        # Packages and bundles from before the geometry model supply the other four graphs.
        settings['onnx_from'] = {name: sha256(origin / f'{name}.onnx') for name in CONTRACTS
                                 if name != 'geometry' or (origin / f'{name}.onnx').is_file()}
        if (origin.parent / 'manifest.json').is_file():
            reuse_manifest = json.loads((origin.parent / 'manifest.json').read_text(encoding='utf-8'))
    # A bundle carries its own validation images; no upstream source download is needed.
    source = None if args.images or bundle else loma_source()
    if args.images:
        images = args.images
    elif bundle:
        images = [origin.parent / name for name in bundle['images']]
    else:
        images = [source / 'assets' / f'0015_{letter}.jpg' for letter in ('A', 'B')]
    images = [path.resolve() for path in images]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.bundle:
        export_bundle(args.bundle.resolve(), shapes, source, images)
        return
    root = args.models.resolve()
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
    if bundle and bundle['version'] == 2:
        references = origin.parent / 'references'
    else:
        references = CACHE / 'references' / sha256(settings_path)
        references.mkdir(parents=True, exist_ok=True)
        # A version 1 bundle stays read-only; its references join the new geometry ones.
        for path in (origin.parent / 'references').glob('*.npz') if bundle else ():
            shutil.copy2(path, references / path.name)
    for directory in ('onnx', 'engines', 'runtime'):
        (pending / directory).mkdir(exist_ok=True)
    copied = set()
    if args.onnx_from:
        origin = args.onnx_from.resolve()
        if (origin / 'onnx').is_dir():
            origin = origin / 'onnx'
        for name in settings['onnx_from']:
            candidate = origin / f'{name}.onnx'
            if name == 'detector' and (reuse_manifest or {}).get('version') == 1:
                check_onnx(candidate, name, reuse_manifest['models'][name]['shapes'])
                print('Upgrade static detector: re-export with dynamic height/width', flush=True)
                continue
            check_onnx(candidate, name, shapes[name])
            destination = pending / 'onnx' / candidate.name
            if not destination.exists():
                shutil.copy2(candidate, destination)
            elif sha256(destination) != settings['onnx_from'][name]:
                raise RuntimeError(f'Unfinished ONNX differs from the requested source: {destination}')
            copied.add(name)
    export_models(pending / 'onnx', shapes, source, images, references)
    # Reuse unchanged engines only from the complete package supplying these ONNX files.
    if reuse_manifest and not same_target(reuse_manifest, settings):
        reuse_manifest = None
    for name in CONTRACTS:
        check_onnx(pending / 'onnx' / f'{name}.onnx', name, shapes[name])
        destination = pending / 'engines' / f'{name}.engine'
        same_precision = name != 'matcher' or matcher_precisions(reuse_manifest or {}) == matcher_precisions(settings)
        if reuse_manifest and name in copied and same_precision and not destination.exists():
            spec = reuse_manifest['models'][name]
            candidate = origin.parent / 'engines' / f'{name}.engine'
            if spec['shapes'] != shapes[name] or sha256(candidate) != spec['sha256']:
                raise RuntimeError(f'Source engine contract/checksum differs: {name}')
            shutil.copy2(candidate, destination)
        build_engine(destination, name, shapes[name], args.matcher_precision, args.hardware_compatibility)
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
    manifest = {'version': 3, 'tensorrt': TRT_VERSION, 'gpu': settings['gpu'],
                'hardware_compatibility': args.hardware_compatibility,
                'matcher_precision': args.matcher_precision,
                'matcher_score_precision': settings['matcher_score_precision'],
                'detector_profile': DETECTOR_PROFILE, 'descriptor_size': [784, 784],
                'max_features': MAX_FEATURES, 'descriptor_dimension': 256,
                'segmentation_mean': [.485, .456, .406], 'segmentation_std': [.229, .224, .225],
                'geometry_size': GEOMETRY_SIZE, 'geometry_tokens': GEOMETRY_TOKENS,
                'geometry_source': {'moge_commit': MOGE_COMMIT, 'model': 'Ruicheng/moge-2-vitl-normal',
                                    'revision': GEOMETRY_REVISION},
                'models': {name: {'inputs': names[0], 'outputs': names[1], 'shapes': shapes[name],
                                 'sha256': sha256(pending / 'engines' / f'{name}.engine')}
                           for name, names in CONTRACTS.items()}}
    write_json(pending / 'manifest.json', manifest)
    # Publish in PowerShell after this process releases any Windows file handles.
    print(f'Model package verified: {pending}', flush=True)


if __name__ == '__main__':
    main()
