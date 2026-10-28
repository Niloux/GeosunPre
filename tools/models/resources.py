"""Fixed source resources; reuse the original models/source cache when present."""
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache' / 'model-prepare'
LOMA_COMMIT = '8fb59c458f2a2ef44314f462467b229c27786bf0'
SEGMENTATION_REVISION = 'de01bae28967510f9ddd496c60a969357195400c'
MOGE_COMMIT = '74fbce054ebed49800de42d0ad0e83495065719a'
GEOMETRY_REVISION = 'cb0e8bbd6b1e243589717c78e750b1ba4c093acf'
TORCH_RESOURCES = {
    'dad.pth': ('https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth',
                '7d827844649faf5213ba97e7adb43e16b936dd3d11ab2d0fc0d602c67fae781f'),
    'loma_B.pt': ('https://github.com/davnords/storage/releases/download/loma/loma_B.pt',
                  '3a38824391e22b33bb3e10377c7736243fd8a2fbf1561446e7e133e497c35758'),
    'dinov2_vitl14_pretrain.pth': (
        'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'),
}
SEGMENTATION_HASHES = {
    'config.json': 'ee7400840fdb1e5045f0b2eba78bf053df8e33a309c4acec31705a48c8cc5c00',
    'preprocessor_config.json': '8039d1d210abaa7117ad78e58cdfd6141a2ec72c03dae891b3cd76737e422c6c',
    'pytorch_model.bin': '187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resource(relative, url, expected):
    cached = CACHE / 'source' / relative
    legacy = ROOT / 'models' / 'source' / relative
    for path in (cached, legacy):
        if path.is_file():
            if sha256(path) != expected:
                raise RuntimeError(f'Source checksum differs: {path}')
            print(f'Reuse source: {path.name}', flush=True)
            return path
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_name(cached.name + '.pending')
    print(f'Download source: {cached.name}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'GeosunPre-model-prepare'})
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as output:
        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if sha256(temporary) != expected:
        raise RuntimeError(f'Download checksum differs: {temporary}')
    temporary.replace(cached)
    return cached


def loma_source():
    archive = resource(Path('LoMa-source.zip'),
                       f'https://codeload.github.com/davnords/LoMa/zip/{LOMA_COMMIT}',
                       '5f0a95982a0fb5eb1cc27259c02ea8dfa54886b86286a1ee8171bc93612a87fe')
    destination = CACHE / 'source' / f'LoMa-{LOMA_COMMIT}'
    ready = destination / '.source-ready'
    if not ready.is_file():
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination.parent)
        ready.write_text(LOMA_COMMIT, encoding='ascii')
    return destination


def moge_source():
    archive = resource(Path(f'MoGe-{MOGE_COMMIT}.zip'),
                       f'https://codeload.github.com/microsoft/MoGe/zip/{MOGE_COMMIT}',
                       'c256a67674c7779ef5e873199353ecff507b9430e4cea5fedfbfce23ceb68035')
    destination = CACHE / 'source' / f'MoGe-{MOGE_COMMIT}'
    ready = destination / '.source-ready'
    if not ready.is_file():
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination.parent)
        ready.write_text(MOGE_COMMIT, encoding='ascii')
    return destination


def geometry_weights():
    return resource(Path('huggingface/hub/models--Ruicheng--moge-2-vitl-normal/snapshots') / GEOMETRY_REVISION / 'model.pt',
                    f'https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/{GEOMETRY_REVISION}/model.pt',
                    '280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01')


def torch_weights():
    return {name: resource(Path('torch/hub/checkpoints') / name, url, digest)
            for name, (url, digest) in TORCH_RESOURCES.items()}


def segmentation_source():
    relative = Path('huggingface/hub/models--nvidia--segformer-b2-finetuned-ade-512-512/snapshots') / SEGMENTATION_REVISION
    paths = [resource(relative / name,
                      f'https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/resolve/{SEGMENTATION_REVISION}/{name}',
                      digest) for name, digest in SEGMENTATION_HASHES.items()]
    # A complete snapshot is required; mixed caches are copied into one local directory.
    if len({path.parent for path in paths}) == 1:
        return paths[0].parent
    destination = CACHE / 'source' / relative
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.parent != destination:
            shutil.copy2(path, destination / path.name)
    return destination
