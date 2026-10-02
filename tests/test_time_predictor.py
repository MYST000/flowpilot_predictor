import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from predictor import NAMES
from predictor.cli import read_config, check_split
from predictor.data import load_split, schema_signature
from predictor.features import extract
from predictor.models import make_model, weighted_quantiles, monotone
from predictor.evaluation import evaluate, metrics, event_order, report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    c = read_config(ROOT/'configs/predictor/default.json')
    c.update(n_estimators=4, min_samples_leaf=2, min_samples=4, min_groups=1,
             n_clusters=2, threads=1, bootstrap_repeats=5)
    return c


@pytest.fixture
def samples():
    out = []
    for i in range(48):
        c = {'backend_id': 'retrieval', 'backend_version': 'v1', 'tool_name': 'search',
             'tool_schema_version': 'search-v1', 'arguments': {'query': 'alpha beta' if i%2 else 'gamma delta'},
             'configured_timeout_ms': 30000, 'batch_index': 0, 'batch_size': 1,
             'execution_mode': 'serial', 'resolution': 'LOCAL_ONLY', 'history': [],
             'load': {}, 'history_snapshot_age_ms': 4, 'load_snapshot_age_ms': 4}
        out.append({'sample_id': str(i), 'task_group_id': 'group'+str(i//4), 'adapter': 'retrieval',
            'source_attempt': 'attempt'+str(i), 'request_id': str(i), 'tool_call_id': str(i),
            'clock_domain': 'controller:test', 'as_of_ns': i*1000, 'dispatch_ns': i*1000+1,
            'observed_ns': i*1000+500, 'predict_seq': 1, 'observe_seq': 3, 'context': c,
            'labels': {'round_trip_ms': float(10+i*i), 'executor_duration_ms': float(i),
                       'execution_status': 'completed'}})
    return out


@pytest.mark.parametrize('algorithm', ['empirical', 'ewma', 'cluster', 'qrf', 'lightgbm'])
def test_quantiles_roundtrip_and_unknown_paths(algorithm, config, samples, tmp_path):
    model = make_model(algorithm, config).fit(samples)
    c = samples[0]['context']
    pred = model.predict(c)
    q = np.array(list(pred['duration_ms'].values()))
    assert list(pred['duration_ms']) == list(NAMES)
    assert np.isfinite(q).all() and (q >= 0).all() and (np.diff(q) >= 0).all()
    joblib.dump(model, tmp_path/'m.joblib')
    assert joblib.load(tmp_path/'m.joblib').predict(c)['duration_ms'] == pred['duration_ms']
    for field, value in [('resolution', 'HISTORICAL_HIT'), ('resolution', 'INFLIGHT_FOLLOWER'),
                         ('resolution', 'UNKNOWN'), ('execution_mode', 'parallel'),
                         ('tool_schema_version', 'unseen'), ('backend_version', 'new')]:
        bad = dict(c); bad[field] = value
        p = model.predict(bad)
        assert p['duration_ms'] is None and p['support']['status'] == 'unsupported'
        assert p['fallback']['reason']


def test_features_do_not_read_outcomes_or_ids(samples):
    c = samples[0]['context']
    original = extract(c)
    poisoned = dict(c, round_trip_ms=999, executor_duration_ms=999, queue_wait_ms=999,
                    timed_out=True, exit_code=99, model_observation='future',
                    effective_timeout_s=99, task_group_id='future', research_split='test')
    assert extract(poisoned) == original
    assert 'top_k' in original and original['top_k_missing'] == 1
    assert 'argument_chars' not in extract(c, 'identity')
    assert not any(k.startswith('history_') for k in extract(c, 'parameters'))


def test_qrf_is_label_distribution_not_tree_mean_distribution(config, samples):
    for s in samples:
        s['context'] = copy.deepcopy(samples[0]['context'])
    config['min_samples_leaf'] = 48
    m = make_model('qrf', config).fit(samples)
    q = m.predict(samples[0]['context'])['duration_ms']
    assert q['q10'] < q['q50'] < q['q99']
    expected = weighted_quantiles(m.y, np.ones(len(samples)))
    np.testing.assert_allclose(list(q.values()), expected)


def test_delayed_feedback_and_state_reset(config, samples):
    m = make_model('ewma', config).fit(samples)
    test = copy.deepcopy(samples[:3])
    for s, t, end in zip(test, [100000, 100010, 100060], [100100, 100050, 100110]):
        s['as_of_ns'], s['dispatch_ns'], s['observed_ns'] = t, t+1, end
        s['labels']['round_trip_ms'] = 99999
    records = evaluate(m, test, online=True)
    versions = {r['sample_id']: r['prediction']['online_state_version'] for r in records}
    assert versions == {'0': 0, '1': 0, '2': 1}
    assert m.state_version == 0
    again = evaluate(m, test, online=True)
    assert [r['prediction'] for r in records] == [r['prediction'] for r in again]
    frozen = evaluate(m, test)
    assert all(r['prediction']['online_state_version'] == 0 for r in frozen)


def test_duplicate_feedback_domain_and_bad_order(samples):
    with pytest.raises(ValueError, match='duplicate'):
        event_order(samples + samples[:1])
    broken = copy.deepcopy(samples)
    broken[0]['clock_domain'] = 'remote'
    with pytest.raises(ValueError, match='clock'):
        event_order(broken)
    broken = copy.deepcopy(samples)
    broken[0]['observed_ns'] = -1
    with pytest.raises(ValueError, match='order'):
        event_order(broken)


def test_conditional_remaining_exhaustion_and_monotonicity(config, samples):
    m = make_model('empirical', config).fit(samples)
    c = samples[0]['context']
    assert m.remaining_from_dispatch(c, 1e9)['duration_ms'] is None
    q = m.remaining_from_dispatch(c, 100)['duration_ms']
    expected = np.quantile(m.y[m.y > 100]-100, [.1, .5, .9, .99])
    np.testing.assert_allclose(list(q.values()), expected)
    np.testing.assert_allclose(monotone([10, -1, 5, 20]), [0, 5, 10, 20])
    with pytest.raises(ValueError):
        monotone([1, 2, 3, np.nan])


def test_q99_scores_group_bootstrap_and_timeout_kept(config, samples):
    m = make_model('empirical', config).fit(samples)
    test = copy.deepcopy(samples[:4])
    test[-1]['labels'].update(round_trip_ms=120000, execution_status='execution_error', normal_completion_right_censored=True)
    records = evaluate(m, test)
    metric = metrics(records)
    assert metric['q99_exceedances'] == 1
    assert metric['coverage']['q99'] == .75
    assert metric['q99_mean_excess_ms'] > 0
    result = report(records, repeats=10, seed=1)
    assert result['bootstrap']['unit'] == 'task_group'
    assert result['by_execution_outcome']['execution_error']['rows'] == 1


def test_test_gate_and_calibration_direction(config, samples):
    m = make_model('empirical', config).fit(samples)
    held = copy.deepcopy(samples[:1]); held[0]['task_group_id'] = 'heldout'
    a = {'model': m, 'calibrated': False}
    with pytest.raises(ValueError, match='allow-test'):
        check_split(a, held, 'test', False)
    check_split(a, held, 'test', True)
    a['calibrated'] = True
    with pytest.raises(ValueError, match='leakage'):
        check_split(a, held, 'tune', False)


def test_existing_prepared_inventory():
    root = ROOT/'runs/predictor_prepared/v1'
    if not (root/'manifest.json').exists():
        pytest.skip('run preparation before integration inventory check')
    groups = []
    for split, count in [('fit',4056), ('tune',694), ('calibration',424), ('test',674)]:
        data = load_split(root, split)
        assert len(data) == count
        groups.append({s['task_group_id'] for s in data})
        for s in data:
            assert s['as_of_ns'] <= s['dispatch_ns'] <= s['observed_ns']
            assert s['context']['history_snapshot_age_ms'] >= 0
            assert 'round_trip_ms' not in extract(s['context'])
    for i,g in enumerate(groups):
        assert not any(g & h for h in groups[i+1:])


def test_schema_ignores_only_random_workspace_location():
    a = {'name': 'code_terminal', 'description': 'Repository: /tmp/flowpilot-code-local-actor-abc/repo.', 'parameters': {'type': 'object'}}
    b = dict(a, description='Repository: /tmp/flowpilot-code-local-actor-xyz/repo.')
    assert schema_signature(a) == schema_signature(b)
    assert schema_signature(a) != schema_signature(dict(b, parameters={'type': 'string'}))


def test_ewma_delayed_residual_uses_prediction_time_center(config, samples):
    from predictor.features import group_key
    m = make_model('ewma', config).fit(samples)
    c = samples[0]['context']; k = group_key(c)
    slow = m.feedback_snapshot(c); fast = m.feedback_snapshot(c)
    original = slow['log_center']
    m.observe(c, 100000, 'fast', prediction_state=fast)
    assert m.means[k] != original
    m.observe(c, 12, 'slow', prediction_state=slow)
    assert m.residuals[k][-1][0] == pytest.approx(np.log1p(12)-original)
    with pytest.raises(ValueError, match='snapshot'):
        m.observe(c, 20, 'missing')


def test_ewma_initialization_respects_prediction_timing(config, samples):
    from predictor.features import group_key
    data = copy.deepcopy(samples[:3])
    for r, start, finish, y in zip(data, [0, 10, 20], [5, 100, 30], [10., 40., 20.]):
        r.update(as_of_ns=start, dispatch_ns=start+1, observed_ns=finish)
        r['labels']['round_trip_ms'] = y
    m = make_model('ewma', config).fit(data)
    residuals = list(m.residuals[group_key(data[0]['context'])])
    np.testing.assert_allclose([x[0] for x in residuals], [np.log1p(20)-np.log1p(10), np.log1p(40)-np.log1p(10)])


def test_sparse_backend_unsupported_but_online_ewma_can_gain_support(config, samples):
    m = make_model('ewma', config).fit(samples[:1])
    c = samples[0]['context']
    assert m.predict(c)['duration_ms'] is None
    for i in range(8):
        snapshot = m.feedback_snapshot(c)
        m.observe(c, float(i+10), f'new-group-{i}', prediction_state=snapshot)
    pred = m.predict(c)
    assert pred['duration_ms'] is not None and pred['method'] == 'ewma'
    assert pred['support']['reference_rows'] == 8
    assert pred['support']['reference_task_groups'] == 8


@pytest.mark.parametrize('weights', [[1., -1.], [1., np.nan], [0., 0.]])
def test_weighted_cdf_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        weighted_quantiles([1., 2.], weights)


def test_calibration_online_mode_is_rejected(config, samples):
    from predictor.cli import validate_mode
    m = make_model('ewma', config).fit(samples)
    a = {'model': m, 'calibrated': True}
    with pytest.raises(ValueError, match='frozen calibration'):
        validate_mode(a, 'online')
    validate_mode(a, 'frozen')


def test_smoke_calibration_cannot_create_full_artifact(config, samples, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import predictor.cli as cli
    m = make_model('empirical', config).fit(samples)
    m.version = 'synthetic-v2'
    held = copy.deepcopy(samples[:8])
    for r in held:
        r['task_group_id'] = 'calibration-' + r['task_group_id']
    a = {'artifact_version': 2, 'model': m, 'offsets': None, 'calibrated': False,
         'smoke': False, 'data_manifest_sha256': 'synthetic'}
    monkeypatch.setattr(cli, 'load_artifact', lambda *args: a)
    monkeypatch.setattr(cli, 'load_split', lambda *args: held)
    model = tmp_path/'parent.joblib'; model.write_bytes(b'synthetic')
    output = tmp_path/'calibrated'
    cli.calibrate(SimpleNamespace(model=model, data=tmp_path, output=output, smoke=True))
    saved = joblib.load(output/'model.joblib')
    manifest = json.loads((output/'manifest.json').read_text())
    assert saved['smoke'] is True
    assert saved['calibration_version'] == manifest['calibration_version']
    assert manifest['evaluation_split'] == 'calibration' and manifest['evaluation_mode'] == 'frozen'


def test_legacy_artifacts_and_code_mismatch_are_rejected(tmp_path):
    from predictor.cli import load_artifact
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'artifact_version': 1}))
    with pytest.raises(ValueError, match='legacy artifact'):
        load_artifact(tmp_path/'model.joblib', tmp_path)
    path.write_text(json.dumps({'artifact_version': 2, 'code_sha256': {}}))
    with pytest.raises(ValueError, match='model/code mismatch'):
        load_artifact(tmp_path/'model.joblib', tmp_path)


def test_prediction_output_names_each_quantile(config, samples):
    m = make_model('empirical', config).fit(samples)
    record = evaluate(m, samples[:1])[0]
    assert record['output_schema_version'] == 2
    assert list(record['score']['pinball_ms']) == list(NAMES)
    assert record['target'] == 'round_trip_ms'
    assert record['quantiles']['q99'] == .99


def test_complete_cli_pipeline_on_synthetic_splits(config, samples, tmp_path):
    from types import SimpleNamespace
    from predictor.cli import train, calibrate, replay
    from predictor.data import digest
    root = tmp_path/'data'; root.mkdir()
    manifest = {'splits': {}}
    for index, split in enumerate(('fit', 'tune', 'calibration', 'test')):
        data = copy.deepcopy(samples[:24])
        for r in data:
            r['split'] = split
            r['sample_id'] = split+'-'+r['sample_id']
            r['task_group_id'] = split+'-'+r['task_group_id']
            for field in ('as_of_ns', 'dispatch_ns', 'observed_ns'):
                r[field] += index*1000000
        path = root/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in data))
        manifest['splits'][split] = {'rows': len(data), 'sha256': digest(path)}
    (root/'manifest.json').write_text(json.dumps(manifest))
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps(config))
    trained = tmp_path/'train'
    train(SimpleNamespace(config=cfg, data=root, output=trained, smoke=True, algorithm='ewma'))
    calibration = tmp_path/'calibration'
    calibrate(SimpleNamespace(model=trained/'model.joblib', data=root, output=calibration, smoke=True))
    for kind, path in [('raw', trained), ('calibrated', calibration)]:
        out = tmp_path/f'test-{kind}'
        replay(SimpleNamespace(model=path/'model.joblib', data=root, output=out,
                               split='test', allow_test=True, mode='frozen', smoke=True))
        metric = json.loads((out/'metrics.json').read_text())
        assert metric['evaluation']['evaluation_split'] == 'test'
        assert metric['evaluation']['smoke_only'] is True
        assert metric['micro']['rows'] == 24
        assert bool(metric['evaluation']['calibration_version']) == (kind == 'calibrated')
