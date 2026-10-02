"""Use the existing experiment profile with the real predictor hooks bound."""

import argparse
import asyncio
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing experiment profile")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cost-model", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate settings and predictor; do not serve")
    args = parser.parse_args()
    from examples.experiments.qwen35_9b_tp4.profile import gateway_settings, load_profile
    profile = load_profile(args.config)
    cost_value = os.getenv("FLOWPILOT_COST_MODEL_PATH")
    cost_path = args.cost_model or (Path(cost_value) if cost_value else None)
    settings = gateway_settings(
        profile, run_dir=args.run_dir.resolve(), registry_path=args.registry,
        api_key=os.environ["FLOWPILOT_INGRESS_API_KEY"],
        dcs_key=os.environ["FLOWPILOT_DCS_ENCRYPTION_KEY"],
        cost_model_path=cost_path,
    )
    if settings.synthetic_tool_duration_enabled:
        raise ValueError("Disable synthetic durations when using the real predictor")
    if args.check:
        from .runtime import PredictorRuntime
        root = Path(__file__).resolve().parents[1]
        config = Path(os.environ.get("FLOWPILOT_PREDICTOR_CONFIG", root / "configs/predictor/runtime.json"))
        async def check():
            async with PredictorRuntime.from_config(config) as runtime:
                print(json.dumps({"status": "ready", "predictor": runtime.snapshot(),
                    "profile": str(args.config), "run_dir": str(args.run_dir),
                    "service_started": False}, ensure_ascii=False, indent=2))
        asyncio.run(check())
        return
    import uvicorn
    from .serve import create_app
    args.run_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=settings.workers)


if __name__ == "__main__":
    main()
