"""Event-ordered evaluation with delayed feedback and task-group bootstrap."""
import copy
import time
from collections import defaultdict
import numpy as np
from predictor import NAMES, QUANTILES, VERSION
from predictor.models import monotone
from predictor.data import require


from predictor.events import event_order


def evaluate(model, samples, online=False, offsets=None, calibration_version=None):
    """Copy initial state per stream; labels are accessed only at observe events."""
    state = copy.deepcopy(model)
    pending, records = {}, []
    for ns, _, _, kind, _, i in event_order(samples):
        s = samples[i]
        if kind == 'predict':
            start = time.perf_counter_ns()
            prediction_state = state.feedback_snapshot(s['context']) if online else None
            prediction = state.predict(s['context'])
            if offsets is not None and prediction['duration_ms'] is not None:
                prediction['duration_ms'] = dict(zip(NAMES, monotone(
                    np.array(list(prediction['duration_ms'].values())) + offsets).tolist()))
                prediction['calibration'] = 'pooled_ms_quantile_residual'
                prediction['calibration_version'] = calibration_version
            cost_ms = (time.perf_counter_ns() - start) / 1e6
            pending[i] = prediction, cost_ms, prediction_state
        else:
            require(i in pending, 'feedback before prediction')
            pred, cost, prediction_state = pending.pop(i)
            y = s['labels'][model.config['target']]
            require(type(y) in (int, float) and np.isfinite(y) and y >= 0, 'invalid feedback label')
            # Score the stored prediction before changing any model state.
            record = {'output_schema_version': 2, 'split': s.get('split', 'synthetic'),
                      'target': model.config['target'], 'quantiles': dict(zip(NAMES, QUANTILES)),
                      'sample_id': s['sample_id'], 'task_group_id': s['task_group_id'],
                      'adapter': s['adapter'], 'tool': s['context']['tool_name'], 'y_ms': y,
                      'execution_status': s['labels']['execution_status'],
                      'prediction': pred, 'predict_ms': cost, 'observed_ns': ns,
                      'score': None,
                      'envelope': {
                          'predictor_schema_version': 3, 'prediction_id': s['sample_id'],
                          'batch_id': s['source_attempt'] + ':' + s['request_id'],
                          'clock_domain': s['clock_domain'], 'as_of_monotonic_ns': s['as_of_ns'],
                          'base_model_version': getattr(model, 'version', VERSION + ':' + model.name), 'method': pred['method'],
                          'online_state_version': pred['online_state_version'], 'calibration_version': calibration_version,
                          'support': pred['support'], 'fallback': pred['fallback'],
                          'valid_for': {'request_id': s['request_id'], 'context_epoch': 0, 'resolution_epoch': 0},
                          'per_call': [{'tool_call_id': s['tool_call_id'],
                              'target': 'round_trip_from_dispatch' if model.config['target'] == 'round_trip_ms' else 'executor_duration',
                              'duration_ms': pred['duration_ms']}],
                      }}
            if pred['duration_ms'] is not None:
                q = np.array([pred['duration_ms'][n] for n in NAMES])
                errors = y - q
                record['score'] = {'pinball_ms': dict(zip(NAMES, np.maximum(np.array(QUANTILES)*errors, (np.array(QUANTILES)-1)*errors).tolist())),
                                   'absolute_q50_error': float(abs(errors[1]))}
            start = time.perf_counter_ns()
            if online:
                state.observe(s['context'], y, s['task_group_id'], prediction_state=prediction_state)
            record['update_ms'] = (time.perf_counter_ns()-start)/1e6
            records.append(record)
    require(not pending, 'unresolved pending')
    return records


def metrics(records):
    valid = [r for r in records if r['score'] is not None]
    out = {'rows': len(records), 'task_groups': len({r['task_group_id'] for r in records}),
           'supported_rows': len(valid), 'unsupported_rows': len(records)-len(valid),
           'fallback_fraction': float(np.mean([r['prediction']['fallback']['used'] for r in records])) if records else None}
    for field in ('predict_ms', 'update_ms'):
        out[field] = dict(zip(('p50', 'p95'), np.quantile([r[field] for r in records], [.5, .95]).tolist())) if records else None
    if not valid:
        return out
    y = np.array([r['y_ms'] for r in valid])
    q = np.array([[r['prediction']['duration_ms'][n] for n in NAMES] for r in valid])
    pinball = np.array([[r['score']['pinball_ms'][n] for n in NAMES] for r in valid]).mean(axis=0)
    out.update(q50_mae_ms=float(np.mean(abs(y-q[:, 1]))), pinball_ms=dict(zip(NAMES, pinball.tolist())),
               mean_pinball_ms=float(pinball.mean()), coverage=dict(zip(NAMES, (y[:, None] <= q).mean(axis=0).tolist())),
               central_80_coverage=float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))),
               central_80_width_ms=float(np.mean(q[:, 2]-q[:, 0])),
               q99_exceedances=int(np.sum(y > q[:, 3])),
               q99_mean_excess_ms=float(np.maximum(y-q[:, 3], 0).mean()),
               q99_expected_exceedances_at_nominal=len(valid)*.01,
               q99_low_support_fraction=float(np.mean([r['prediction']['support']['q99_low_support'] for r in valid])))
    return out


def report(records, repeats=500, seed=0):
    by_tool, by_adapter, by_outcome = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        by_tool[r['adapter'] + '/' + r['tool']].append(r)
        by_adapter[r['adapter']].append(r)
        by_outcome[r['execution_status']].append(r)
    tool_metrics = {k: metrics(v) for k, v in by_tool.items()}
    supported = [v for v in tool_metrics.values() if v['supported_rows']]
    macro = {n: float(np.mean([v['pinball_ms'][n] for v in supported])) for n in NAMES} if supported else None
    result = {'micro': metrics(records), 'by_tool': tool_metrics,
              'by_adapter': {k: metrics(v) for k, v in by_adapter.items()},
              'by_execution_outcome': {k: metrics(v) for k, v in by_outcome.items()},
              'tool_macro_pinball_ms': macro,
              'tools_total': len(tool_metrics), 'tools_scored': len(supported),
              'macro_excludes_unsupported_tools': len(supported) != len(tool_metrics),
              'tool_macro_mean_pinball_ms': float(np.mean(list(macro.values()))) if macro else None,
              'q99_note': 'Exploratory: few independent groups and expected tail observations; no strict coverage guarantee.',
              'latency_note': 'Includes context feature extraction and model prediction; excludes input JSON parsing and scheduler/RPC.',
              'bootstrap': None}
    if repeats > 0:
        # Cluster bootstrap: resample whole task groups, preserve all calls in each group.
        groups = sorted({r['task_group_id'] for r in records})
        tools = sorted(by_tool)
        gi, ti = {g:i for i,g in enumerate(groups)}, {t:i for i,t in enumerate(tools)}
        sums = np.zeros((len(groups), len(tools), 4)); counts = np.zeros((len(groups), len(tools)))
        coverage_sums = np.zeros((len(groups), len(tools)))
        for r in records:
            if r['score'] is not None:
                g, t = gi[r['task_group_id']], ti[r['adapter']+'/'+r['tool']]
                sums[g,t] += [r['score']['pinball_ms'][name] for name in NAMES]; counts[g,t] += 1
                coverage_sums[g,t] += r['y_ms'] <= r['prediction']['duration_ms']['q99']
        rng = np.random.default_rng(seed)
        boot, coverage_boot = [], []
        for _ in range(repeats):
            indices = rng.integers(0, len(groups), len(groups))
            n = counts[indices].sum(axis=0); s = sums[indices].sum(axis=0)
            if (n > 0).any():
                boot.append((s[n > 0]/n[n > 0, None]).mean(axis=0))
                coverage_boot.append(float(coverage_sums[indices].sum()/n.sum()))
        if boot:
            bounds = np.quantile(boot, [.025, .975], axis=0)
            result['bootstrap'] = {'unit': 'task_group', 'repeats': repeats,
                                   'micro_q99_coverage_95_interval': np.quantile(coverage_boot, [.025, .975]).tolist(),
                                   'tool_macro_pinball_95_interval': {name: bounds[:,j].tolist() for j,name in enumerate(NAMES)}}
    return result
