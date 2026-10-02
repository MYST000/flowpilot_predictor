#!/usr/bin/env python3
"""Wait for one fixed-config suite, snapshot its models, then calibrate/test once."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ('empirical', 'ewma', 'cluster', 'qrf', 'lightgbm')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


class Status:
    def __init__(self, directory, run_dir, output):
        self.directory = directory
        self.value = {'started_at_utc': now(), 'run_dir': str(run_dir), 'output': str(output),
                      'selection_policy': 'fixed configuration supplied by user; no automatic tuning'}

    def update(self, state, **extra):
        self.value.update(state=state, updated_at_utc=now(), **extra)
        write_json(self.directory/'state.json', self.value)
        line = f'{self.value["updated_at_utc"]} {state} {json.dumps(extra, ensure_ascii=False)}'
        print(line, flush=True)
        with (self.directory/'watcher.log').open('a') as stream:
            stream.write(line+'\n')


def wait_for_training(run_dir, poll_seconds, max_wait_seconds):
    deadline = time.monotonic() + max_wait_seconds
    marker = run_dir/'exit_code'
    while True:
        if marker.exists():
            value = marker.read_text().strip()
            # An empty marker can briefly be visible during shell redirection.
            if value:
                try:
                    code = int(value)
                except ValueError as exc:
                    raise RuntimeError('Invalid training exit_code marker') from exc
                if code != 0:
                    raise RuntimeError(f'Training suite failed with exit_code={code}; final evaluation was not started')
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Training completion marker did not arrive before max-wait-seconds')
        time.sleep(min(poll_seconds, remaining))


def snapshot_models(run_dir, data, destination):
    """Validate every model before copying; no calibration/test labels are read."""
    sys.path.insert(0, str(ROOT))
    from predictor.cli import load_artifact
    metadata = {}
    for algorithm in ALGORITHMS:
        folder = run_dir/algorithm
        a = load_artifact(folder/'model.joblib', data)
        manifest = json.loads((folder/'manifest.json').read_text())
        if (manifest.get('status') != 'complete' or a['smoke'] or a['calibrated'] or
                a['model'].name != algorithm or manifest.get('training_split') != 'fit'):
            raise RuntimeError(f'{algorithm}: expected a completed, full, uncalibrated fit model')
        metadata[algorithm] = {
            'model_sha256': manifest['model_sha256'], 'manifest_sha256': sha256(folder/'manifest.json'),
            'version': manifest['version'], 'config': manifest['config'],
        }
    destination.mkdir(exist_ok=False)
    for algorithm, expected in metadata.items():
        folder = destination/algorithm
        folder.mkdir()
        for name in ('model.joblib', 'manifest.json'):
            shutil.copyfile(run_dir/algorithm/name, folder/name)
        if (sha256(folder/'model.joblib') != expected['model_sha256'] or
                sha256(folder/'manifest.json') != expected['manifest_sha256']):
            raise RuntimeError(f'{algorithm}: source changed while snapshotting; final evaluation was not started')
    return metadata


def execute_finalizer(models, output, data, log_path, status):
    command = ['bash', str(ROOT/'scripts/finalize_predictor_suite.sh'), 'full',
               str(models), str(output), str(data), '--allow-test']
    env = dict(os.environ, PREDICTOR_PYTHON=sys.executable)
    with log_path.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status.update('finalizing', child_pid=process.pid, final_log=str(log_path))
            return process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--data', type=Path, default=ROOT/'runs/predictor_prepared/v1')
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--max-wait-seconds', type=float, default=86400)
    p.add_argument('--parameters-frozen', action='store_true', help='Use these configurations as final; do not tune/select on test')
    p.add_argument('--allow-test', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Print plan only; no waiting or files created')
    p.add_argument('--validate-only', action='store_true', help=argparse.SUPPRESS)
    return p


def run(args):
    run_dir, data = args.run_dir.resolve(), args.data.resolve()
    output = args.output.resolve() if args.output else Path(str(run_dir)+'_final')
    job_dir = Path(str(output)+'.autofinalize')
    if not 0 < args.poll_seconds <= 60 or not 0 < args.max_wait_seconds <= 7*86400:
        raise ValueError('poll-seconds must be in (0,60], max-wait-seconds in (0,604800]')
    if not (data/'manifest.json').is_file():
        raise ValueError('Prepare the dataset first')
    if output.exists() or job_dir.exists():
        raise ValueError('Output or auto-finalize control directory already exists; refusing duplicate execution')
    if not args.dry_run and not (args.parameters_frozen and args.allow_test):
        raise ValueError('Automatic final evaluation requires --parameters-frozen --allow-test')
    plan = {'run_dir': str(run_dir), 'output': str(output), 'control_directory': str(job_dir),
            'data': str(data), 'wait_for': str(run_dir/'exit_code'),
            'sequence': ['wait for exit_code=0', 'validate and snapshot five full fit models',
                         'calibration', 'raw frozen test', 'calibrated frozen test', 'summary.csv'],
            'automatic_hyperparameter_search': False}
    if args.dry_run or args.validate_only:
        print(json.dumps(plan, indent=2))
        return 0
    # Exclusive mkdir reserves this output even if two watcher sessions race.
    job_dir.mkdir(parents=True, exist_ok=False)
    status = Status(job_dir, run_dir, output)
    result = 1
    try:
        status.update('waiting_for_training', max_wait_seconds=args.max_wait_seconds)
        wait_for_training(run_dir, args.poll_seconds, args.max_wait_seconds)
        status.update('validating_models')
        selection = snapshot_models(run_dir, data, job_dir/'selected_models')
        write_json(job_dir/'selection.json', {'frozen_at_utc': now(), 'models': selection,
                   'data_manifest_sha256': sha256(data/'manifest.json'),
                   'finalizer_sha256': sha256(ROOT/'scripts/finalize_predictor_suite.sh')})
        code = execute_finalizer(job_dir/'selected_models', output, data, job_dir/'finalize.log', status)
        if code != 0:
            raise RuntimeError(f'Final evaluation failed with exit_code={code}; inspect finalize.log and per-stage logs')
        if (output/'exit_code').read_text().strip() != '0' or not (output/'summary.csv').is_file():
            raise RuntimeError('Final process exited without successful completion artifacts')
        status.update('complete', summary=str(output/'summary.csv'))
        result = 0
    except (KeyboardInterrupt, InterruptedError):
        status.update('cancelled', error='Watcher interrupted; any owned finalizer was terminated')
        result = 130
    except Exception as exc:
        status.update('failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        (job_dir/'exit_code').write_text(str(result)+'\n')
    return result


def interrupted(signum, frame):
    raise InterruptedError(f'signal {signum}')


def main():
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        return run(parser().parse_args())
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
