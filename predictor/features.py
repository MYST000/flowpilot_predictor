"""Only explicitly allowed context fields reach estimators."""
import json
import math
import re
import numpy as np

LEVELS = ('identity', 'parameters', 'history_state')


def group_key(c):
    return (c['backend_id'], c['backend_version'], c['tool_name'], c['tool_schema_version'])


def backend_key(c):
    return (c['backend_id'], c['backend_version'])


def argument_text(c):
    return json.dumps(c['arguments'], ensure_ascii=False, sort_keys=True)


def extract(c, level='history_state'):
    if level not in LEVELS:
        raise ValueError(level)
    f = {k: str(c[k]) for k in ('backend_id', 'backend_version', 'tool_name', 'tool_schema_version')}
    if level == 'identity':
        return f
    def number(name, value):
        valid = type(value) in (int, float) and math.isfinite(value)
        f[name] = float(value) if valid else 0.0
        f[name + '_missing'] = float(not valid)
    a = c['arguments']
    text = argument_text(c)
    query = str(a.get('query', ''))
    words = re.findall(r'\w+', query)
    command = str(a.get('command', ''))
    for name, value in {
        'argument_chars': len(text), 'query_chars': len(query), 'query_words': len(words),
        'query_unique_words': len(set(words)), 'command_chars': len(command),
        'code_chars': len(str(a.get('file_text', a.get('content', a.get('code', ''))))),
        'pipe_count': command.count('|'), 'redirect_count': command.count('>') + command.count('<'),
        'command_separators': command.count(';') + command.count('&&') + command.count('\n'),
        'top_k': a.get('top_k'), 'max_chars': a.get('max_chars'), 'max_lines': a.get('max_lines'),
        'configured_timeout_ms': c.get('configured_timeout_ms'),
        'batch_index': c.get('batch_index'), 'batch_size': c.get('batch_size'),
    }.items():
        number(name, value)
    # Fixed categories avoid memorizing paths, queries or task identifiers.
    first = command.strip().split(maxsplit=1)[0] if command.strip() else ''
    f['command_category'] = first if first in {'python', 'python3', 'cat', 'ls', 'bash', 'echo', 'timeout', 'pytest'} else 'other'
    op = str(a.get('command', a.get('operation', '')))
    f['edit_operation'] = op if op in {'view', 'create', 'str_replace', 'insert', 'undo_edit'} else 'other'
    if level == 'parameters':
        return f
    h = c.get('history', [])
    values = [x['rtt_ms'] for x in h]
    ewma = None
    for value in values:
        ewma = value if ewma is None else .2 * value + .8 * ewma
    for name, value in {
        'history_count': len(values), 'history_last': values[-1] if values else None,
        'history_mean': float(np.mean(values)) if values else None, 'history_ewma': ewma,
        'history_failure_rate': float(np.mean([x['failed'] for x in h])) if h else None,
        'history_snapshot_age_ms': c.get('history_snapshot_age_ms'),
        'load_snapshot_age_ms': c.get('load_snapshot_age_ms'),
        'remaining_budget_t0_s': c.get('remaining_budget_t0_s'),
    }.items():
        number(name, value)
    for name in ('tool_inflight', 'llm_inflight', 'active_sessions', 'max_sessions'):
        number('t0_' + name, c.get('load', {}).get(name))
    return f
