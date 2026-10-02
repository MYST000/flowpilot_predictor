"""Run the pinned native BrowseComp launcher with out-of-band timing metadata."""

import os
import runpy
from pathlib import Path

from browsecomp_timing import install_timing


def main():
    root = Path(os.environ.get("FLOWPILOT_ROOT", Path(__file__).resolve().parents[1]))
    namespace = runpy.run_path(str(root / "runtime/browsecomp-native/serve.py"))
    original_class = namespace["FastMCP"]

    def timed_server(*args, **kwargs):
        return install_timing(original_class(*args, **kwargs))

    entry = namespace["main"]
    entry.__globals__["FastMCP"] = timed_server
    entry()


if __name__ == "__main__":
    main()
