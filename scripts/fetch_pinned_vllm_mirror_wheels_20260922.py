"""One-off cached-wheel fetch helper, with official PyPI SHA256 verification.

The uv metadata cache also contains rejected resolver candidates. These files
must be supplied through ``uv pip install --find-links`` so dependency
constraints select the final versions; do not install a glob of all wheels as
direct requirements. The installed package lock is the final authority.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.parse
import urllib.request

from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

ROOT = Path("/home/qiulin/flowpilot_predictor")
OUT = ROOT / "downloads" / "vllm_pinned_dependency_wheels_20260922"
CACHE = ROOT / "cache" / "uv" / "wheels-v6" / "pypi"
LOG = ROOT / "evidence/preparation/vllm_install_20260922.cdn_direct.log"
TAGSET = set(sys_tags())


def candidate(name: str) -> dict:
    for meta in sorted(
        (CACHE / name).glob("*.msgpack"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        matches = re.findall(
            rb'https://files\.pythonhosted\.org/[^\s\x00-\x20"<>]+?\.whl',
            meta.read_bytes(),
        )
        for match in matches:
            url = match.decode()
            filename = urllib.parse.unquote(url.rsplit("/", 1)[1])
            _name, version, _build, tags = parse_wheel_filename(filename)
            if not TAGSET.intersection(tags):
                continue
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/{name}/{version}/json", timeout=60
            ) as response:
                info = json.load(response)
            exact = next(item for item in info["urls"] if item["filename"] == filename)
            return {
                "name": name,
                "version": str(version),
                "filename": filename,
                "official_url": exact["url"],
                "expected_sha256": exact["digests"]["sha256"],
                "expected_bytes": exact["size"],
                "uv_metadata_file": str(meta),
            }
    raise RuntimeError(f"No selected compatible wheel metadata: {name}")


def fetch(name: str) -> dict:
    item = candidate(name)
    target = OUT / item["filename"]
    if not target.exists():
        mirror = item["official_url"].replace(
            "https://files.pythonhosted.org/", "https://mirrors.ustc.edu.cn/pypi/", 1
        )
        part = target.with_suffix(target.suffix + ".partial")
        env = os.environ.copy()
        env["NO_PROXY"] = env["no_proxy"] = "*"
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "2",
                "--max-time",
                "300",
                "--continue-at",
                "-",
                "--output",
                str(part),
                mirror,
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(f"{name}: {result.stderr}")
        actual = hashlib.sha256(part.read_bytes()).hexdigest()
        if (
            actual != item["expected_sha256"]
            or part.stat().st_size != item["expected_bytes"]
        ):
            raise RuntimeError(
                f"{name}: mirror SHA256 or size does not match official PyPI metadata"
            )
        part.rename(target)
        item["download_url"] = mirror
    else:
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["expected_sha256"]:
            raise RuntimeError(f"{name}: previously saved wheel SHA256 mismatch")
    item["sha256_verified"] = True
    item["path"] = str(target)
    print(json.dumps({"verified": name, "bytes": item["expected_bytes"]}), flush=True)
    return item


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = LOG.read_text().splitlines()
    pending = set()
    for line in lines:
        if line.startswith("Downloading "):
            pending.add(line.split()[1])
        elif line.startswith(" Downloaded "):
            pending.discard(line.split()[1])
    pending = {name for name in pending if (CACHE / name).exists()}
    print(json.dumps({"pending": sorted(pending)}), flush=True)
    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        jobs = {executor.submit(fetch, name): name for name in sorted(pending)}
        for future in concurrent.futures.as_completed(jobs):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"name": jobs[future], "error": str(error)})
                print(json.dumps(failures[-1]), flush=True)
            (OUT / "official_sha256_manifest.json").write_text(
                json.dumps({"wheels": results, "failures": failures}, indent=2) + "\n"
            )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
