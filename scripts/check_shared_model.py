"""Validate the shared 27B artifact and run four synthetic prediction requests."""
import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowpilot_predictor_bridge import CallIdentity, PredictorRuntime, ToolPredictionRequest


async def check(config_path):
    async with PredictorRuntime.from_config(config_path, project_root=ROOT) as runtime:
        contexts = []
        for key, indices in runtime.models[0].groups.items():
            context = copy.deepcopy(runtime.models[0].contexts[indices[0]])
            context['arguments'] = {'query': 'predictor installation check', 'top_k': 5}
            context['history'] = []
            contexts.append(context)
        if not contexts:
            raise RuntimeError('No supported tools in the shared model')
        identities = []
        for i in range(4):
            identity = CallIdentity(f'install-check-{i}', 'line', f'request-{i}',
                                    'tail', 'llm', f'tool-{i}', 1, 0, 0)
            runtime.submit(ToolPredictionRequest(identity, contexts[i % len(contexts)]))
            identities.append(identity)
        predictions = await asyncio.gather(*(runtime.wait(identity) for identity in identities))
        for prediction in predictions:
            if prediction.get('duration_ms') is None:
                raise RuntimeError('Prediction check failed: ' + json.dumps(prediction))
        print(json.dumps({'status': 'ready', 'model_version': runtime.model_version,
            'model_sha256': runtime.artifact_sha256, 'workers': len(runtime.models),
            'selected_quantile': runtime.quantile, 'online': runtime.online,
            'online_method': runtime.online_method,
            'predictions': [{'tool': contexts[i % len(contexts)]['tool_name'],
                             'duration_ms': prediction['duration_ms'],
                             'duration_estimate_ms': prediction['duration_estimate_ms']}
                            for i, prediction in enumerate(predictions)],
            'benchmark_started': False, 'gateway_started': False,
            'duration_sink_bound': runtime.sink is not None}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT/'configs/predictor/shared-data1.example.json')
    args = parser.parse_args()
    asyncio.run(check(args.config))


if __name__ == '__main__':
    main()
