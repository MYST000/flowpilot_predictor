"""Deterministic replay order shared by initialization and evaluation."""
from predictor.data import require


def event_order(samples):
    require(len({s['sample_id'] for s in samples}) == len(samples), 'duplicate feedback/sample id')
    require(len({s['clock_domain'] for s in samples}) == 1, 'cannot merge clock domains')
    events = []
    for i, s in enumerate(samples):
        require(s['as_of_ns'] <= s['dispatch_ns'] <= s['observed_ns'], 'invalid event order')
        require((s['as_of_ns'], s['predict_seq']) < (s['observed_ns'], s['observe_seq']),
                'feedback must follow prediction')
        for kind, ns, seq in [('predict', s['as_of_ns'], s['predict_seq']), ('observe', s['observed_ns'], s['observe_seq'])]:
            events.append((ns, s['source_attempt'], seq, kind, s['sample_id'], i))
    return sorted(events)
