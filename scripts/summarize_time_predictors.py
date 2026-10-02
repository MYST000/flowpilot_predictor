#!/usr/bin/env python3
"""Collect completed predictor metrics; never fit models or read raw labels."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    records = []
    for path in sorted(args.root.glob('*/metrics.json')):
        manifest = json.loads((path.parent/'manifest.json').read_text())
        if manifest.get('status') != 'complete':
            continue
        m = json.loads(path.read_text()); micro = m['micro']
        record = {'experiment': path.parent.name, 'algorithm': manifest.get('algorithm'), 'target': manifest.get('target'),
                  'training_split': manifest.get('training_split'),
                  'evaluation_split': manifest.get('evaluation_split'),
                  'evaluation_mode': manifest.get('evaluation_mode'),
                  'model_version': manifest.get('version'),
                  'calibration_version': manifest.get('calibration_version'),
                  'smoke_only': manifest.get('smoke_only'),
                  'rows': micro['rows'], 'supported_rows': micro['supported_rows'],
                  'unsupported_rows': micro['unsupported_rows'],
                  'tools_total': m.get('tools_total'), 'tools_scored': m.get('tools_scored'),
                  'tool_macro_mean_pinball_ms': m['tool_macro_mean_pinball_ms'],
                  'q50_mae_ms': micro.get('q50_mae_ms'),
                  'q99_exceedances': micro.get('q99_exceedances'),
                  'q99_mean_excess_ms': micro.get('q99_mean_excess_ms'),
                  'central_80_coverage': micro.get('central_80_coverage'),
                  'central_80_width_ms': micro.get('central_80_width_ms'),
                  'fallback_fraction': micro['fallback_fraction'],
                  'predict_p95_ms': micro['predict_ms']['p95']}
        for q in ('q10', 'q50', 'q90', 'q99'):
            record[f'{q}_macro_pinball_ms'] = (m['tool_macro_pinball_ms'] or {}).get(q)
            record[f'{q}_micro_coverage'] = micro.get('coverage', {}).get(q)
        records.append(record)
    if not records:
        raise SystemExit('No completed metrics found')
    output = args.output or args.root/'summary.csv'
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(json.dumps({'experiments': len(records), 'output': str(output)}))


if __name__ == '__main__':
    main()
