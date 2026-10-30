"""Fixed source resources; reuse the original models/source cache when present."""
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache' / 'model-prepare'
LOMA_COMMIT = '8fb59c458f2a2ef44314f462467b229c27786bf0'
MOGE_COMMIT = '74fbce054ebed49800de42d0ad0e83495065719a'
GEOMETRY_REVISION = 'cb0e8bbd6b1e243589717c78e750b1ba4c093acf'
# facebook/sam3 is gated on Hugging Face, so the checkpoint is supplied as a local file.
GROUNDING_SHA256 = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
TORCH_RESOURCES = {
    'dad.pth': ('https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth',
                '7d827844649faf5213ba97e7adb43e16b936dd3d11ab2d0fc0d602c67fae781f'),
    'loma_B.pt': ('https://github.com/davnords/storage/releases/download/loma/loma_B.pt',
                  '3a38824391e22b33bb3e10377c7736243fd8a2fbf1561446e7e133e497c35758'),
    'dinov2_vitl14_pretrain.pth': (
        'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'),
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


def grounding_weights(path):
    """The pinned SAM3 checkpoint; a supplied file is copied into the source cache for later runs."""
    cached = CACHE / 'source' / 'sam3.pt'
    if cached.is_file():
        if sha256(cached) != GROUNDING_SHA256:
            raise RuntimeError(f'Source checksum differs: {cached}')
        print(f'Reuse source: {cached.name}', flush=True)
        return cached
    if not path or not Path(path).is_file():
        raise RuntimeError('Model packages include SAM3 grounding: pass -Sam3 sam3.pt (facebook/sam3, gated on '
                           'Hugging Face) once')
    if sha256(path) != GROUNDING_SHA256:
        raise RuntimeError(f'SAM3 checkpoint checksum differs: {path}')
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_name(cached.name + '.pending')
    shutil.copyfile(path, temporary)
    temporary.replace(cached)
    return cached


def torch_weights():
    return {name: resource(Path('torch/hub/checkpoints') / name, url, digest)
            for name, (url, digest) in TORCH_RESOURCES.items()}
