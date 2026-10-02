"""Launch with uvicorn flowpilot_predictor_bridge.serve:create_app --factory.

The launcher opts into the real T1 predictor; scheduling/reuse remain configured
by the existing FLOWPILOT_* settings. No forecast or synthetic prior is enabled.
"""

import json
import os
from pathlib import Path


def create_app(settings=None):
    from flowpilot.app import create_app as framework_app
    from flowpilot.config import Settings
    from .framework import FrameworkPredictor
    from .runtime import PredictorRuntime

    root = Path(__file__).resolve().parents[1]
    path = Path(
        os.environ.get(
            "FLOWPILOT_PREDICTOR_CONFIG", root / "configs/predictor/runtime.json"
        )
    )
    config = json.loads(path.read_text())
    for key in ("model_path", "prepared_data"):
        config[key] = str(root / config[key])
    if config.get("artifact_code_root"):
        config["artifact_code_root"] = str(root / config["artifact_code_root"])
    settings = settings if settings is not None else Settings.from_env()
    if settings.synthetic_tool_duration_enabled:
        raise ValueError(
            "Disable synthetic duration when enabling the real RTT predictor"
        )
    return framework_app(
        settings,
        tool_duration_adapter=FrameworkPredictor(
            PredictorRuntime.from_artifact(**config)
        ),
    )
