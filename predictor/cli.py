"""Explicit preparation, training, calibration and evaluation; test is opt-in."""
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from predictor import NAMES, QUANTILES, VERSION, ARTIFACT_VERSION
from predictor.data import prepare, load_split, write_json, digest, require, signature
from predictor.models import ALGORITHMS, make_model
from predictor.evaluation import evaluate, report

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    c = json.loads(Path(path).read_text())
    require(c['target'] in ('round_trip_ms', 'executor_duration_ms'), 'unsupported target')
    require(c['target_transform'] in ('raw', 'log1p'), 'unsupported transform')
    require(c['feature_level'] in ('identity', 'parameters', 'history_state'), 'unsupported feature level')
    for k in ('threads', 'min_samples', 'min_groups', 'n_estimators', 'min_samples_leaf', 'max_depth', 'n_clusters', 'max_tfidf_features', 'residual_buffer'):
        require(type(c[k]) is int and c[k] > 0, f'invalid {k}')
    require(0 < c['ewma_alpha'] <= 1 and c['learning_rate'] > 0, 'invalid learning rate')
    require(c['bootstrap_repeats'] >= 0, 'invalid bootstrap count')
    return c


def versions():
    return {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in
            ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl')}}


def save_records(output, records):
    with (output/'predictions.jsonl').open('w') as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')


def subset(samples, limit):
    # Smoke only: a chronological prefix, no outcome-dependent sampling.
    return sorted(samples, key=lambda s: (s['as_of_ns'], s['source_attempt'], s['predict_seq'], s['sample_id']))[:limit]


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'predictor').glob('*.py'))}


def output_context(model, split, mode, smoke, calibration_version=None):
    return {'algorithm': model.name, 'target': model.config['target'],
            'training_split': 'fit', 'evaluation_split': split, 'evaluation_mode': mode,
            'version': model.version, 'calibration_version': calibration_version,
            'smoke_only': smoke}


def validate_mode(artifact, mode):
    require(mode != 'online' or artifact['model'].name == 'ewma',
            'online adaptation is implemented for EWMA; other models remain frozen')
    require(not artifact['calibrated'] or mode == 'frozen',
            'frozen calibration cannot be applied to an adapting online model')


def train(args):
    config = read_config(args.config)
    if args.smoke:
        config.update(n_estimators=8, min_samples_leaf=2, min_samples=4, min_groups=1,
                      n_clusters=2, threads=1, bootstrap_repeats=10)
    samples = load_split(args.data, 'fit')
    evaluation = load_split(args.data, 'tune')
    if args.smoke:
        samples, evaluation = subset(samples, 120), subset(evaluation, 36)
    require(not {s['task_group_id'] for s in samples} & {s['task_group_id'] for s in evaluation}, 'fit/tune overlap')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=config['threads']):
        begin = time.perf_counter()
        model = make_model(args.algorithm, config).fit(samples)
        fit_seconds = time.perf_counter() - begin
        model.version = VERSION + ':' + args.algorithm + ':' + signature({
            'config': config, 'data': digest(Path(args.data)/'manifest.json'),
            'code': code_hashes(), 'smoke': args.smoke})[:16]
        records = evaluate(model, evaluation)
        summary = report(records, config['bootstrap_repeats'], config['seed'])
    # Serialize ONLY the fit model, not tune-adapted state.
    summary['evaluation'] = output_context(model, 'tune', 'frozen', args.smoke)
    artifact = {'artifact_version': ARTIFACT_VERSION, 'model': model, 'offsets': None, 'calibrated': False,
                'data_manifest_sha256': digest(Path(args.data)/'manifest.json'),
                'smoke': args.smoke, 'calibration_version': None, 'fit_end_ns': max(s['observed_ns'] for s in samples),
                'fit_clock_domain': samples[0]['clock_domain']}
    joblib.dump(artifact, output/'model.joblib')
    save_records(output, records)
    write_json(output/'metrics.json', summary)
    manifest = {'status': 'complete', 'artifact_version': ARTIFACT_VERSION,
                **output_context(model, 'tune', 'frozen', args.smoke),
                'quantiles': list(QUANTILES), 'config': config, 'smoke_only': args.smoke,
                'training_split': 'fit', 'evaluation_split': 'tune', 'evaluation_mode': 'frozen',
                'training_rows': len(samples), 'evaluation_rows': len(evaluation),
                'fit_seconds': fit_seconds, 'dependencies': versions(), 'code_sha256': code_hashes(),
                'data_manifest_sha256': artifact['data_manifest_sha256'],
                'model_sha256': digest(output/'model.joblib'),
                'supported_paths': ['serial/execute'], 'ready_prediction_implemented': False,
                'calibration': None}
    write_json(output/'manifest.json', manifest)
    print(json.dumps({'output': str(output), 'status': 'complete', 'smoke_only': args.smoke}))


def load_artifact(model_path, data):
    model_path = Path(model_path)
    manifest = json.loads((model_path.parent/'manifest.json').read_text())
    require(manifest.get('artifact_version') == ARTIFACT_VERSION,
            'legacy artifact: retrain with the reviewed v2 code; prior smoke artifacts are not reusable')
    require(manifest.get('code_sha256') == code_hashes(), 'model/code mismatch: retrain or use the exact saved code version')
    require(digest(model_path) == manifest['model_sha256'], 'model hash mismatch')
    a = joblib.load(model_path)  # Local artifacts only; pickle is not an untrusted interchange format.
    require(a['data_manifest_sha256'] == digest(Path(data)/'manifest.json'), 'data manifest mismatch')
    return a


def check_split(a, samples, split, allow_test):
    require(split != 'test' or allow_test, 'test requires --allow-test after freezing the protocol')
    require(not a['calibrated'] or split == 'test', 'calibrated artifact is reserved for final test; prevents calibration -> tune leakage')
    fit_groups = {s['task_group_id'] for s in a['model'].samples}
    require(not fit_groups & {s['task_group_id'] for s in samples}, 'fit/evaluation group overlap')


def replay(args):
    a = load_artifact(args.model, args.data)
    samples = load_split(args.data, args.split)
    check_split(a, samples, args.split, args.allow_test)
    require(not a['smoke'] or args.smoke, 'smoke artifact cannot produce a full evaluation')
    if args.smoke:
        samples = subset(samples, 36)
    model = a['model']
    validate_mode(a, args.mode)
    if args.mode == 'online':
        require(all(s['clock_domain'] == a['fit_clock_domain'] and s['as_of_ns'] > a['fit_end_ns'] for s in samples),
                'online replay requires fit to precede evaluation in the same clock domain')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=model.config['threads']):
        records = evaluate(model, samples, online=args.mode == 'online', offsets=a['offsets'],
                           calibration_version=a['calibration_version'])
        summary = report(records, 10 if args.smoke else model.config['bootstrap_repeats'], model.config['seed'])
    summary['evaluation'] = output_context(model, args.split, args.mode, args.smoke, a['calibration_version'])
    save_records(output, records)
    write_json(output/'metrics.json', summary)
    write_json(output/'manifest.json', {'status': 'complete',
               **summary['evaluation'], 'config': model.config, 'artifact_version': ARTIFACT_VERSION,
               'smoke_only': args.smoke, 'model_sha256': digest(args.model), 'rows': len(samples),
               'calibrated': a['calibrated'], 'data_manifest_sha256': a['data_manifest_sha256'],
               'code_sha256': code_hashes(), 'dependencies': versions(),
               'state_reset': True, 'ordering': 'monotonic_ns, source_attempt, source_seq, event_kind, sample_id'})
    print(json.dumps({'output': str(output), 'status': 'complete'}))


def calibrate(args):
    a = load_artifact(args.model, args.data)
    require(not a['calibrated'], 'already calibrated')
    require(not a['smoke'] or args.smoke, 'smoke artifact requires --smoke')
    samples = load_split(args.data, 'calibration')
    check_split(a, samples, 'calibration', False)
    if args.smoke:
        samples = subset(samples, 36)
    with threadpool_limits(limits=a['model'].config['threads']):
        records = evaluate(a['model'], samples)
    valid = [r for r in records if r['score'] is not None]
    require(bool(valid), 'no supported calibration samples')
    residuals = np.array([[r['y_ms']-r['prediction']['duration_ms'][n] for n in NAMES] for r in valid])
    offsets = np.array([np.quantile(residuals[:, j], q) for j,q in enumerate(QUANTILES)])
    a.update(offsets=offsets, calibrated=True, smoke=a['smoke'] or args.smoke,
             calibration_version='calibration:' + signature({
                 'base_version': a['model'].version, 'offsets_ms': offsets.tolist(),
                 'samples': [r['sample_id'] for r in valid], 'method': 'pooled_frozen_ms_v1'})[:16])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    joblib.dump(a, output/'model.joblib')
    write_json(output/'manifest.json', {'status': 'complete', 'artifact_version': ARTIFACT_VERSION,
        **output_context(a['model'], 'calibration', 'frozen', a['smoke'], a['calibration_version']),
        'code_sha256': code_hashes(), 'dependencies': versions(), 'config': a['model'].config,
        'model_sha256': digest(output/'model.joblib'),
        'parent_model_sha256': digest(args.model), 'data_manifest_sha256': a['data_manifest_sha256'],
        'calibration_split': 'calibration', 'rows': len(valid), 'task_groups': len({r['task_group_id'] for r in valid}),
        'smoke_only': args.smoke, 'offsets_ms': dict(zip(NAMES, offsets.tolist())),
        'method': 'pooled marginal quantile residual; monotone rearrangement after correction',
        'guarantee': 'empirical only; no strict finite-sample/conditional coverage guarantee'})
    print(json.dumps({'output': str(output), 'status': 'complete'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--project', default=str(ROOT)); p.add_argument('--output', required=True)
    p = commands.add_parser('train')
    p.add_argument('--algorithm', choices=ALGORITHMS, required=True)
    p.add_argument('--config', default=str(ROOT/'configs/predictor/default.json'))
    for name in ('data', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--smoke', action='store_true')
    for command in ('evaluate', 'calibrate'):
        p = commands.add_parser(command)
        for name in ('data', 'model', 'output'):
            p.add_argument('--'+name, required=True)
        p.add_argument('--smoke', action='store_true')
        if command == 'evaluate':
            p.add_argument('--split', choices=('tune', 'calibration', 'test'), default='tune')
            p.add_argument('--mode', choices=('frozen', 'online'), default='frozen')
            p.add_argument('--allow-test', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.project, args.output)
    elif args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        replay(args)
    else:
        calibrate(args)


if __name__ == '__main__':
    main()
