"""The historical two-segment local reconstruction runner."""
import json
from pathlib import Path

import pycolmap

from ..local_reconstruction import (STAGES, database_and_seed, mapper_options,
                                    assert_holdout_isolated, cheirality_health,
                                    bounded_bundle_adjustment)
from ..local_evaluation import evaluate_reconstruction
from ..scene import digest, file_digest, write_json
from .local_data import load_local
from .local_evaluation import evaluate_triangle_protocol, write_experiment_report

def run_local_reconstruction(work):
    out,scene,data,features=load_local(work)
    records={im['name']:im for im in scene['images']}
    reports={}
    for label,segment in data['segments'].items():
        directory=out/label
        images=[records[name] for name in segment['image_names']]
        split=json.loads(Path(segment['split_path']).read_text())
        contract={'geometry_pipeline_version':1,'scene':scene['fingerprint'],'frontend_config':data['config'],
                  'features':{name:data['features'][name]['token'] for name in segment['image_names']},
                  'pairs':[{k:e[k] for k in ('a','b','token')} for e in segment['pairs']],
                  'split_sha256':file_digest(segment['split_path'])}
        contract['fingerprint']=digest(contract)
        contract_path=directory/'run_contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text())['fingerprint']!=contract['fingerprint']:
            raise ValueError(f'{label}: scene/features/split changed; use a fresh result directory to preserve the previous experiment')
        write_json(contract_path,contract)
        baseline_path=directory/'mapping'
        if not (baseline_path/'cameras.bin').exists():
            db,seed=directory/'database.db',directory/'seed'
            if not (db.exists() and (seed/'cameras.bin').exists()):
                db,seed=database_and_seed(directory,scene,images,segment['pairs'],features,split)
            opts=mapper_options()
            maps=pycolmap.incremental_mapping(db,Path(scene['input_root'])/'Image',directory/'mapper_output',
                                              options=opts,input_path=seed)
            if not maps:
                raise RuntimeError(f'{label}: no local reconstruction')
            rec=max(maps.values(),key=lambda r:r.num_reg_images())
            baseline_path.mkdir(exist_ok=True);rec.write(baseline_path)
            write_json(directory/'mapping_summary.json',{'registered':rec.num_reg_images(),'selected':len(images),
                'points':rec.num_points3D(),'observations':rec.compute_num_observations(),
                'pose_priors':False,'intrinsics_fixed_during_mapping':True,'options':opts.summary()})
        else:
            rec=pycolmap.Reconstruction(baseline_path)
        assert_holdout_isolated(rec,split)
        initial={cid:cam.params.copy() for cid,cam in rec.cameras.items()}
        stage_reports={}
        for stage in STAGES:
            stage_dir=directory/'stages'/stage
            if (stage_dir/'ba.json').exists():
                rec=pycolmap.Reconstruction(stage_dir/'sparse/0')
                ba=json.loads((stage_dir/'ba.json').read_text())
                health=cheirality_health(rec)
                if health['invalid_depth']:
                    raise RuntimeError(f'Cached stage {label}/{stage} has invalid depth: {health}')
                if 'cheirality' not in ba:
                    ba['cheirality']=health
                    write_json(stage_dir/'ba.json',ba)
            else:
                print(f'{label}: BA {stage}',flush=True)
                ba=bounded_bundle_adjustment(rec,stage,initial)
                if stage=='principal' and ba['termination_type'].endswith('NO_CONVERGENCE'):
                    # One explicit continuation from an iteration-limited
                    # candidate. Keep the first result and never use heldout
                    # errors to decide whether to continue optimization.
                    archive=stage_dir.with_name(stage+'_guarded_400_iterations')
                    archived_model=archive/'sparse/0';archived_model.mkdir(parents=True,exist_ok=True)
                    rec.write(archived_model);write_json(archive/'ba.json',ba)
                    previous=ba
                    ba=bounded_bundle_adjustment(rec,stage,initial)
                    ba['continuation']={'from':str(archive),'previous_solver':previous['solver'],
                        'previous_elapsed_seconds':previous['elapsed_seconds'],
                        'reason':'Continue the same candidate from the previous iteration limit, without using validation errors.'}
                model=stage_dir/'sparse/0';model.mkdir(parents=True,exist_ok=True)
                rec.write(model);rec.write_text(model)
                write_json(stage_dir/'ba.json',ba)
            assert_holdout_isolated(rec,split)
            validation=evaluate_reconstruction(rec,scene,features,split,stage_dir)
            stage_reports[stage]={'ba':ba,'validation':validation}
        reports[label]={'selected_images':len(images),'registered_images':rec.num_reg_images(),
            'points':rec.num_points3D(),'stages':stage_reports,'split_tracks':len(split['tracks'])}
        write_json(directory/'comparison.json',reports[label])
    write_json(out/'comparison.json',reports)
    evaluate_triangle_protocol(out,scene,data,features)
    write_experiment_report(out,scene,reports)
    return reports
