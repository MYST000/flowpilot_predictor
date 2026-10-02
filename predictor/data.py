"""Prepare explicit T1 contexts; outcomes live in a separate labels object."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

SPLITS = ('fit', 'tune', 'calibration', 'test')
EXPECTED_RTT = dict(zip(SPLITS, (4056, 694, 424, 674)))
EXPECTED_READY = dict(zip(SPLITS, (3768, 645, 401, 613)))
ENV_TOOLS = {
    'hotpot': {'search', 'read_document'},
    'browsecomp': {'search', 'get_document'},
    'livecodebench': {'code_terminal', 'code_file_editor'},
}


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def schema_signature(schema):
    # Runtime workspace locations do not change tool semantics. Preserve all
    # other description/parameter details and retain raw snapshot file hashes.
    frozen = dict(schema)
    if 'description' in frozen:
        frozen['description'] = re.sub(
            r'/tmp/flowpilot-code-local-actor-[^/\s]+/repo',
            '<TASK_REPOSITORY>', frozen['description'])
    return signature(frozen)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def require(ok, message):
    if not ok:
        raise ValueError(message)


def key(r):
    return r['source_attempt'], r['request_id']


def index_unique(items, key_fn):
    result = {}
    for r in items:
        k = key_fn(r)
        require(k not in result, f'duplicate key: {k}')
        result[k] = r
    return result


def prepare(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'stage': 'T1_llm_response_proxy', 'splits': {},
                'source_sha256': {}, 'quantiles': [.1, .5, .9, .99],
                'limitations': ['T1 uses llm_response before parsing, not a fresh parse-complete snapshot',
                                'load and per-task history are observed T0 snapshots with ages',
                                'execute/serial only; no cache hit/follower or ready predictor',
                                'dataset_revision and frozen environment identity proxy corpus version',
                                'tool schema frozen at the first request snapshot of each attempt; random local actor repo path normalized']}
    groups_by_split = {}
    def record(path):
        manifest['source_sha256'][str(Path(path).relative_to(project))] = digest(path)
    for split in SPLITS:
        root = project / 'runs/campaigns' / f'mixed_c4_v1_{split}'
        source = [root / p for p in ('campaign.json', 'prediction_dataset/inputs.jsonl',
                  'prediction_dataset/targets.jsonl', 'tool_timing_dataset/tools.jsonl')]
        for p in source:
            record(p)
        inputs = index_unique(rows(source[1]), key)
        targets = index_unique(rows(source[2]), key)
        require(inputs.keys() == targets.keys(), 'unmatched input/target')
        tools = list(rows(source[3]))
        index_unique(tools, lambda r: (*key(r), r['call']['tool_call_id']))
        attempts = {}
        for inp in inputs.values():
            attempt = inp['source_attempt']
            require(inp['research_split'] == split, 'split mismatch')
            if attempt in attempts:
                continue
            event_path = Path(attempt) / 'events.jsonl'
            record(event_path)
            events = [e for e in rows(event_path) if e['event'] in
                      ('llm_response', 'tool_start', 'tool_end', 'tool_error')]
            responses = index_unique((e for e in events if e['event'] == 'llm_response'), lambda e: e['request_id'])
            starts = index_unique((e for e in events if e['event'] == 'tool_start'), lambda e: e['tool_call_id'])
            ends = index_unique((e for e in events if e['event'] in ('tool_end', 'tool_error')), lambda e: e['tool_call_id'])
            snapshot = Path(attempt) / inp['features']['request_snapshot']['path']
            record(snapshot)
            blob = json.loads(snapshot.read_text())
            schema = {t['function']['name']: t['function'] for t in blob['tools']}
            attempts[attempt] = responses, starts, ends, schema
        counts, skipped = Counter(), Counter()
        groups = set()
        out_path = output / f'{split}.jsonl'
        ready_path = output / f'{split}.ready_labels.jsonl'
        with out_path.open('w') as stream:
            for row in tools:
                require(row['research_split'] == split, 'tool split mismatch')
                inp, target = inputs[key(row)], targets[key(row)]
                label, call = row['labels'], row['call']
                require(row['task_group_id'] == inp['task_group_id'] == target['task_group_id'], 'group mismatch')
                if not label['executed']:
                    skipped['not_executed'] += 1
                    continue
                require(label['execution_status'] in ('completed', 'execution_error'), 'unobservable end')
                require(finite(label['round_trip_ms']), 'invalid RTT')
                require(label['arguments_json_valid'] and isinstance(call['arguments_parsed'], dict), 'invalid arguments')
                responses, starts, ends, schemas = attempts[row['source_attempt']]
                response, start, end = responses[row['request_id']], starts[call['tool_call_id']], ends[call['tool_call_id']]
                domain = inp['clock_domain']
                require(all(e['clock_domain'] == domain for e in (response, start, end)), 'clock domain mismatch')
                t1, begin, finish = (e['monotonic_ns'] for e in (response, start, end))
                require(t1 <= begin <= finish, 'invalid lifecycle')
                require(begin == row['tool_start_monotonic_ns'] and finish == row['tool_end_monotonic_ns'], 'event/export mismatch')
                f = inp['features']
                t0 = f['monotonic_ns']
                require(t0 <= t1, 'future snapshot')
                adapter, tool = row['adapter'], call['tool_name']
                require(tool in ENV_TOOLS[adapter] and tool in schemas, 'unsupported tool schema')
                profile, env = f['tool_execution_profile'], f['environment_known_at_t0']
                require(profile['intra_task_tool_concurrency'] == 1, 'only serial supported')
                history = []
                for h in f['prior_tool_executions']:
                    prior_end = ends[h['tool_call_id']]
                    require(prior_end['clock_domain'] == domain and prior_end['monotonic_ns'] <= t0, 'future history')
                    if h['tool_name'] == tool and finite(h.get('round_trip_ms')):
                        history.append({'rtt_ms': h['round_trip_ms'], 'failed': h['execution_status'] != 'completed'})
                load = f.get('t0_features', {}).get('client_load', {})
                sampled = load.get('sampled_monotonic_ns')
                require(sampled is None or sampled <= t0, 'future load')
                args = dict(call['arguments_parsed'])
                for name, prop in schemas[tool].get('parameters', {}).get('properties', {}).items():
                    if name not in args and 'default' in prop:
                        args[name] = prop['default']
                actions = target['labels']['actions']
                # Count proposals in the complete response, never successful executions.
                batch_count = sum(a['tool_name'] in ENV_TOOLS[adapter] for a in actions)
                schema_version = schema_signature(schemas[tool])
                backend = env.get('backend', adapter)
                backend_version = signature({'environment': env, 'dataset_revision': inp['dataset_revision'],
                                             'actor': inp['replica_id'], 'profile': profile})
                context = {
                    'backend_id': backend, 'backend_version': backend_version,
                    'tool_schema_version': schema_version, 'tool_name': tool,
                    'arguments': args, 'configured_timeout_ms': profile['tool_timeout_seconds'] * 1000,
                    'batch_index': call['batch_index'], 'batch_size': batch_count,
                    'execution_mode': 'serial', 'resolution': 'LOCAL_ONLY',
                    'history': history[-64:], 'history_snapshot_age_ms': (t1 - t0) / 1e6,
                    'load': {k: load.get(k) for k in ('tool_inflight', 'llm_inflight', 'active_sessions', 'max_sessions')},
                    'load_snapshot_age_ms': None if sampled is None else (t1 - sampled) / 1e6,
                    'remaining_budget_t0_s': f['budget_at_t0'].get('remaining_seconds'),
                }
                sample = {
                    'sample_id': signature([*key(row), call['tool_call_id']]),
                    'source_attempt': row['source_attempt'], 'request_id': row['request_id'],
                    'tool_call_id': call['tool_call_id'], 'task_group_id': row['task_group_id'],
                    'split': split, 'adapter': adapter, 'clock_domain': domain,
                    'as_of_ns': t1, 'predict_seq': response['seq'], 'dispatch_ns': begin,
                    'observed_ns': finish, 'observe_seq': end['seq'], 'context': context,
                    'labels': {'round_trip_ms': label['round_trip_ms'],
                               'executor_duration_ms': label.get('executor_duration_ms'),
                               'execution_status': label['execution_status'],
                               'normal_completion_right_censored': label['normal_completion_right_censored']},
                }
                stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + '\n')
                groups.add(row['task_group_id'])
                counts[f'{adapter}/{tool}'] += 1
        ready_count = 0
        with ready_path.open('w') as stream:
            for target in targets.values():
                l, m = target['labels'], target['masks']
                if (m['environment_batch_duration'] and m['complete_arguments'] and
                    l['has_unknown_tool_call'] is False and l['next_request_observed'] and
                    finite(l['next_request_prepared_gap_ms'])):
                    stream.write(json.dumps({'source_attempt': target['source_attempt'],
                        'request_id': target['request_id'], 'task_group_id': target['task_group_id'],
                        'next_request_prepared_gap_ms': l['next_request_prepared_gap_ms']}) + '\n')
                    ready_count += 1
        n = sum(counts.values())
        require(n == EXPECTED_RTT[split] and ready_count == EXPECTED_READY[split], f'{split}: handoff counts changed')
        groups_by_split[split] = groups
        manifest['splits'][split] = {'rows': n, 'task_groups': len(groups), 'by_tool': dict(counts),
                                    'skipped': dict(skipped), 'ready_rows': ready_count,
                                    'sha256': digest(out_path), 'ready_sha256': digest(ready_path)}
        print(f'prepared {split}: RTT={n}, ready={ready_count}', flush=True)
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            require(not groups_by_split[a] & groups_by_split[b], f'group leakage: {a}/{b}')
    manifest['feature_contract'] = 'predictor.features.extract(context): identity / parameters / history_state'
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_split(directory, split):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    path = root / f'{split}.jsonl'
    require(digest(path) == manifest['splits'][split]['sha256'], f'{split}: prepared data hash mismatch')
    result = list(rows(path))
    require(len(result) == manifest['splits'][split]['rows'], 'prepared count mismatch')
    return result
