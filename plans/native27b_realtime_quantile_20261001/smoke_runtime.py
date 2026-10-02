"""Real 27B model bridge smoke; recorded feedback, no tool/LLM execution."""

import asyncio
import copy
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/flowpilot_predictor')
sys.path.insert(0, str(ROOT))

import numpy as np

from flowpilot_predictor_bridge import CallIdentity, ToolFeedback, ToolPredictionRequest
from flowpilot_predictor_bridge.runtime import PredictorRuntime
from predictor.data import digest


async def run():
    data = Path('/data1/ql_flowpilot_predictor/predictor_prepared/native27b_1077_v1')
    with (data / 'test.jsonl').open() as f:
        samples = [json.loads(line) for line in f]
    search = [s for s in samples if s['adapter'] == 'hotpot' and s['context']['tool_name'] == 'search'][:80]
    others = OrderedDict()
    for s in samples:
        key = (s['adapter'], s['context']['tool_name'])
        if key != ('hotpot', 'search'):
            others.setdefault(key, s)
    selected = search + list(others.values())
    handed_off, results, latencies = [], [], []
    async def sink(prior):
        handed_off.append(prior)
        return True
    config = ROOT / 'configs/predictor/runtime.json'
    async with PredictorRuntime.from_config(config, duration_sink=sink) as runtime:
        maximum_pending = 0
        for start in range(0, len(selected), 4):
            batch = selected[start:start + 4]
            requests = []
            for offset, s in enumerate(batch, start):
                identity = CallIdentity(s['task_group_id'], 'line', 'request-' + str(offset),
                    'tail-' + str(offset), 'llm-' + str(offset), s['tool_call_id'], 1, 0, offset + 1)
                c = copy.deepcopy(s['context'])
                c['task_group_id'] = s['task_group_id']
                requests.append(ToolPredictionRequest(identity, c))
            began = time.perf_counter_ns()
            runtime.on_llm_response(requests)
            maximum_pending = max(maximum_pending, runtime.snapshot()['native_or_queued'])
            predictions = await asyncio.gather(*(runtime.wait(req.identity) for req in requests))
            elapsed = (time.perf_counter_ns() - began) / 1e6
            latencies.append(elapsed)
            assert all(p['duration_ms'] is not None for p in predictions), predictions
            for req in requests:
                runtime.resolve(req.identity, 'LOCAL_ONLY', version=1)
            await runtime.drain_deliveries()
            delivered = {p.identity: p for p in handed_off}
            for req, s, prediction in zip(requests, batch, predictions):
                prior = delivered[req.identity]
                assert prior.duration_estimate_ms == prediction['duration_ms']['q50']
                assert prior.duration_p90 == prediction['duration_ms']['q90']
                update = runtime.observe(ToolFeedback(req.identity, 'feedback:' + req.identity.request_id,
                    1, 'LOCAL_ONLY', True, s['labels']['round_trip_ms'], status=s['labels']['execution_status']))
                assert update['status'] == 'updated'
                assert runtime.observe(ToolFeedback(req.identity, 'feedback:' + req.identity.request_id,
                    1, 'LOCAL_ONLY', True, s['labels']['round_trip_ms'], status=s['labels']['execution_status']))['status'] == 'duplicate'
                results.append({'tool': s['adapter'] + '/' + s['context']['tool_name'],
                    'state_version_at_prediction': prediction['online_state_version'],
                    'raw_q50_ms': prediction['raw_duration_ms']['q50'],
                    'delivered_q50_ms': prior.duration_estimate_ms,
                    'active_quantiles': prediction['online_calibration']['active_quantiles']})
        probe_context = copy.deepcopy(search[0]['context'])
        probe_context['task_group_id'] = search[0]['task_group_id']
        identity = CallIdentity('probe', 'line', 'probe-request', 'probe-tail', 'probe-llm', 'probe-tool', 1, 0, 999)
        probe = await runtime.predict(ToolPredictionRequest(identity, probe_context))
        assert probe['online_state_version'] == 80
        assert 'q50' in probe['online_calibration']['active_quantiles']
        assert all(model.state_version == 0 for model in runtime.models)
        assert runtime.snapshot()['metrics']['online_updates'] == len(selected)
        summary = {'checked_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
            'scope': 'Actual 27B model and runtime; recorded RTT feedback; in-process duration sink fixture; no vLLM/LLM/tool/scheduler execution',
            'config_sha256': digest(config), 'runtime': runtime.snapshot(),
            'prediction_requests': len(selected) + 1, 'measured_feedback_replayed': len(selected),
            'accepted_q50_handoffs': len(handed_off), 'maximum_pending_predictions': maximum_pending,
            'batch_completion_latency_ms': {'p50': float(np.quantile(latencies, .5)), 'p95': float(np.quantile(latencies, .95))},
            'probe_after_80_feedbacks': {'state_version': probe['online_state_version'],
                'raw_duration_ms': probe['raw_duration_ms'], 'online_duration_ms': probe['duration_ms'],
                'calibration': probe['online_calibration']}, 'calls': results}
    output = Path('/data1/ql_flowpilot_predictor/predictor_experiments/native27b_realtime_quantile_v1/runtime_smoke.json')
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['calls', 'probe_after_80_feedbacks']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(run())
