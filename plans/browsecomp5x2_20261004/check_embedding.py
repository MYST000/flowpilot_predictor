"""Load and exercise the configured semantic embedding on CPU only."""

import asyncio
import json
import math
import sys
from pathlib import Path

from flowpilot.reuse.semantic import Qwen3Embedding


async def main():
    root = Path(sys.argv[1])
    profile = json.loads((root / "profile.json").read_text())
    embedder = Qwen3Embedding(
        model_path=profile["flowpilot"]["reuse_embedding_model_path"]
    )
    vectors = await embedder.embed(["Mount Everest"])
    assert len(vectors) == 1 and len(vectors[0]) == 1024
    assert math.isclose(sum(v * v for v in vectors[0]), 1.0, abs_tol=1e-4)
    device = str(next(embedder._model.parameters()).device)
    assert device == "cpu"
    report = {
        "status": "passed",
        "device": device,
        "dimension": len(vectors[0]),
        "index_id": embedder.index_id,
    }
    (root / "embedding-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    asyncio.run(main())
