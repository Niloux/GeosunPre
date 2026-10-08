"""Invoke the single production workflow and resume completed stages."""

from importlib import import_module
from pathlib import Path

from geosunpre.artifacts import digest_cache, file_digest, write_json
from geosunpre.config import CONFIG_FILE, PipelineConfig, load_config, save_config
from geosunpre.pipeline import STAGES
from geosunpre.pipeline.preflight import preflight
from geosunpre.pipeline.runtime import StageRecorder, work_lock
from geosunpre.pipeline.task import bind_task, configure_offline
from geosunpre.scene import load_scene, prepare

HANDLERS = {
    'masks': ('data', 'prepare_masks'),
    'features': ('data', 'extract_features'),
    'matches': ('data', 'match_features'),
    'observations': ('observations', 'freeze_observations'),
    'calibration': ('calibration', 'calibrate'),
    'finalize': ('finalization', 'finalize'),
    'register': ('registration', 'register'),
}


def run_pipeline(work, *, input_root=None, config=None, stage='all', output=None,
                 image_mode=None, include_masks=None, include_ply=None, ply_voxel_size=None, model_cache=None):
    """Omitted input, config and delivery options inherit the frozen task."""
    if stage not in (*STAGES, 'all'):
        raise ValueError(f'Unknown pipeline stage: {stage}')
    work = Path(work).expanduser().resolve()
    selected = STAGES if stage == 'all' else (stage,)
    if config is not None and not isinstance(config, PipelineConfig):
        config = PipelineConfig.from_file(config)
    if input_root is not None:
        input_root = Path(input_root).expanduser().resolve()
        if work.is_relative_to(input_root) or input_root.is_relative_to(work):
            raise ValueError('Choose a work directory separate from the raw input')
    configure_offline()
    with work_lock(work), digest_cache():
        if (work / CONFIG_FILE).exists():
            saved = load_config(work)
            if config is not None:
                save_config(work, config)
            config = saved
        else:
            if any(path.name != '.pipeline.lock' for path in work.iterdir()):
                raise ValueError('Unmanaged work directory; choose a fresh --work directory')
            if 'prepare' not in selected or input_root is None:
                raise ValueError('A new run requires --input and the prepare/all stage')
            config = config or PipelineConfig()
            save_config(work, config)
        if (work / 'manifest.json').exists():
            saved_input = Path(load_scene(work)['input_root']).resolve()
            if input_root is not None and input_root != saved_input:
                raise ValueError('Input directory changed; use a fresh work directory')
            input_root = saved_input
        if 'prepare' in selected and input_root is None:
            raise ValueError('--input is required until preparation completes')
        task, delivery, model_cache = bind_task(
            work, output=output, image_mode=image_mode, include_masks=include_masks,
            include_ply=include_ply, ply_voxel_size=ply_voxel_size, model_cache=model_cache)
        with StageRecorder(work, selected) as recorder:
            summary = {'work': str(work), 'stages': [], 'config_sha256': file_digest(work / CONFIG_FILE),
                       'task_fingerprint': task['fingerprint'], 'delivery_complete': False,
                       'invocation_id': recorder.value['invocation_id']}
            if 'prepare' not in selected:
                with recorder.stage('preflight'):
                    summary['preflight'] = preflight(work, selected, delivery, model_cache=model_cache)
            for current in selected:
                with recorder.stage(current):
                    if current == 'prepare':
                        scene = prepare(input_root, work)
                        calibration = config.calibration_images(scene)
                        summary['input'] = {'images': len(scene['images']),
                                            'calibration_images': len(calibration)}
                        if len(selected) > 1:
                            summary['preflight'] = preflight(work, selected, delivery, model_cache=model_cache)
                    elif current == 'sfm':
                        from geosunpre.pipeline.reconstruction import build_mapping, optimize
                        build_mapping(work)
                        optimize(work)
                    elif current == 'export':
                        from geosunpre.pipeline.registration import export_registered
                        bind_task(work, **delivery, model_cache=model_cache)
                        summary['export'] = export_registered(work, **delivery)
                        summary['delivery_complete'] = True
                    else:
                        module, function = HANDLERS[current]
                        handler = getattr(import_module('geosunpre.pipeline.' + module), function)
                        result = (handler(work, model_cache=model_cache) if current == 'masks'
                                  else handler(work))
                        if current in ('finalize', 'register'):
                            summary['model' if current == 'finalize' else 'registration'] = result
                    summary['stages'].append(current)
            timing = recorder.timing()
            summary.update(status='completed', timing=timing)
            write_json(work / 'run_result.json', summary)
            recorder.complete(timing)
            return summary
