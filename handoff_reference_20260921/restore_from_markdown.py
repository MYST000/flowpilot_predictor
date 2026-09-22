"""Restore FlowPilot text files from the marked Markdown appendix; execute no payload."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


def read_records(markdown):
    # Universal newlines allow a Windows editor to save the Markdown as CRLF.
    lines = markdown.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    prefix = "<!-- FLOWPILOT_TEXT_FILE_V1 "
    ending = " -->\n"
    records = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(prefix):
            i += 1
            continue
        if not line.endswith(ending):
            raise ValueError("Malformed file marker")
        meta = json.loads(line[len(prefix):-len(ending)])
        name = meta["path"]
        if not isinstance(name, str) or "\\" in name or "\x00" in name:
            raise ValueError("Invalid relative path")
        rel = PurePosixPath(name)
        if not rel.parts or rel.is_absolute() or ".." in rel.parts or rel.as_posix() != name:
            raise ValueError("Unsafe/noncanonical path: " + name)
        if name in records:
            raise ValueError("Duplicate path: " + name)
        opening = re.fullmatch(r"(`{8,})[a-zA-Z0-9_+-]*\n", lines[i + 1])
        if opening is None:
            raise ValueError("Missing source fence: " + name)
        fence = opening.group(1) + "\n"
        j = i + 2
        while j < len(lines) and lines[j] != fence:
            j += 1
        if j + 1 >= len(lines) or lines[j + 1] != "<!-- FLOWPILOT_TEXT_FILE_END -->\n":
            raise ValueError("Missing source end: " + name)
        value = "".join(lines[i + 2:j])
        if meta.get("display_added_final_newline") is True:
            if not value.endswith("\n"):
                raise ValueError("Missing display newline: " + name)
            value = value[:-1]
        raw = value.encode("utf-8")
        if len(raw) != meta["bytes"] or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
            raise ValueError("Content/hash mismatch: " + name)
        mode = meta["mode"]
        if mode not in {"0644", "0755"}:
            raise ValueError("Unexpected mode: " + name)
        records[name] = (raw, int(mode, 8))
        i = j + 2
    if "SHA256SUMS" not in records:
        raise ValueError("Missing embedded SHA256SUMS")
    expected = {}
    for line in records["SHA256SUMS"][0].decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or match.group(2) in expected:
            raise ValueError("Invalid/duplicate manifest entry")
        expected[match.group(2)] = match.group(1)
    if set(expected) != set(records) - {"SHA256SUMS"}:
        raise ValueError("Manifest/file set mismatch")
    for name, digest in expected.items():
        if hashlib.sha256(records[name][0]).hexdigest() != digest:
            raise ValueError("Manifest checksum mismatch: " + name)
    return records


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 restore_from_markdown.py INPUT.md NEW_OUTPUT_DIRECTORY")
    markdown = Path(sys.argv[1]).expanduser()
    target = Path(os.path.abspath(os.path.expanduser(sys.argv[2])))
    if target.exists() or target.is_symlink():
        raise SystemExit("Refusing existing output directory: " + str(target))
    records = read_records(markdown)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700)  # Exclusive creation: never merge into an existing run.
    for name, (raw, mode) in records.items():
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write(raw)
        dest.chmod(mode)
    print("Restored and verified", len(records), "files to", target)
    print("Next: cd into this directory and run sha256sum -c SHA256SUMS")


if __name__ == "__main__":
    main()
