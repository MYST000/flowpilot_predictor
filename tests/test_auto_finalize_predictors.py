import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('auto_finalize', Path(__file__).resolve().parents[1]/'scripts/auto_finalize_predictors.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


def args(tmp_path):
    data = tmp_path/'data'; data.mkdir(); (data/'manifest.json').write_text('{}')
    train = tmp_path/'train'; train.mkdir()
    return SimpleNamespace(run_dir=train, output=tmp_path/'final', data=data,
                           poll_seconds=.001, max_wait_seconds=.01,
                           parameters_frozen=True, allow_test=True, dry_run=False, validate_only=False)


def test_dry_run_creates_no_output(tmp_path):
    a = args(tmp_path); a.dry_run=True; a.allow_test=False; a.parameters_frozen=False
    assert auto.run(a)==0
    assert not a.output.exists() and not Path(str(a.output)+'.autofinalize').exists()


def test_explicit_frozen_protocol_required(tmp_path):
    a=args(tmp_path);a.parameters_frozen=False
    with pytest.raises(ValueError, match='parameters-frozen'):
        auto.run(a)
    assert not a.output.exists()


def test_failed_training_does_not_start_finalizer(tmp_path, monkeypatch):
    a=args(tmp_path);(a.run_dir/'exit_code').write_text('1\n')
    monkeypatch.setattr(auto, 'snapshot_models', lambda *a: pytest.fail('must not snapshot failed training'))
    assert auto.run(a)==1
    control=Path(str(a.output)+'.autofinalize')
    assert json.loads((control/'state.json').read_text())['state']=='failed'
    assert not a.output.exists()


def test_missing_completion_times_out(tmp_path):
    a=args(tmp_path)
    assert auto.run(a)==1
    state=json.loads((Path(str(a.output)+'.autofinalize')/'state.json').read_text())
    assert 'TimeoutError' in state['error'] and not a.output.exists()


def test_training_finishes_after_wait_begins(tmp_path, monkeypatch):
    a=args(tmp_path)
    monkeypatch.setattr(auto.time, 'sleep', lambda _: (a.run_dir/'exit_code').write_text('0\n'))
    auto.wait_for_training(a.run_dir, .001, .5)


def test_successful_one_shot_workflow_without_real_models_or_test(tmp_path, monkeypatch):
    a=args(tmp_path);(a.run_dir/'exit_code').write_text('0\n')
    def snapshot(run, data, destination):
        destination.mkdir()
        return {'synthetic': {'config': {}}}
    def finalize(models, output, data, log, status):
        assert models.is_dir()
        status.update('finalizing')
        output.mkdir();(output/'exit_code').write_text('0\n');(output/'summary.csv').write_text('synthetic\n')
        log.write_text('synthetic finalizer\n')
        return 0
    monkeypatch.setattr(auto, 'snapshot_models', snapshot)
    monkeypatch.setattr(auto, 'execute_finalizer', finalize)
    assert auto.run(a)==0
    control=Path(str(a.output)+'.autofinalize')
    assert json.loads((control/'state.json').read_text())['state']=='complete'
    assert (control/'selection.json').exists()
    with pytest.raises(ValueError, match='duplicate'):
        auto.run(a)


def test_finalizer_failure_is_recorded(tmp_path, monkeypatch):
    a=args(tmp_path);(a.run_dir/'exit_code').write_text('0\n')
    monkeypatch.setattr(auto, 'snapshot_models', lambda *a: {})
    monkeypatch.setattr(auto, 'execute_finalizer', lambda *a: 7)
    assert auto.run(a)==1
    control=Path(str(a.output)+'.autofinalize')
    assert 'exit_code=7' in json.loads((control/'state.json').read_text())['error']
    assert (control/'exit_code').read_text().strip()=='1'
