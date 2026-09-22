"""Resume official large files with bounded parallel HTTP ranges and final hashes."""
import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--size', required=True, type=int)
    ap.add_argument('--hash', required=True)
    ap.add_argument('--algorithm', choices=['sha256', 'md5'], default='sha256')
    ap.add_argument('--resolve-via-proxy', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    direct = urllib.request.build_opener()
    url = args.url
    if args.resolve_via_proxy:
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            url = r.url
    parts = args.output.with_name(args.output.name + '.parts')
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 4 * 1024 * 1024
    ranges = [(i, start, min(start + chunk_size, args.size) - 1)
              for i, start in enumerate(range(0, args.size, chunk_size))]
    prefix = args.output.with_name(args.output.name + '.part')
    if prefix.exists():
        with prefix.open('rb') as src:
            for i, start, end in ranges:
                size = end - start + 1
                if end >= prefix.stat().st_size:
                    break
                dst = parts / f'{i:06d}.chunk'
                if not dst.exists():
                    src.seek(start)
                    dst.write_bytes(src.read(size))
    def fetch(item):
        i, start, end = item
        dst = parts / f'{i:06d}.chunk'
        size = end - start + 1
        if dst.exists() and dst.stat().st_size == size:
            return size
        tmp = dst.with_suffix('.part')
        for attempt in range(8):
            try:
                offset = tmp.stat().st_size if tmp.exists() else 0
                if offset > size:
                    raise ValueError('oversized range part')
                if offset < size:
                    req = urllib.request.Request(url, headers={'Range': f'bytes={start+offset}-{end}'})
                    with direct.open(req, timeout=60) as r:
                        expected = f'bytes {start+offset}-{end}/{args.size}'
                        if r.status != 206 or r.headers.get('Content-Range') != expected:
                            raise ValueError('server did not honor exact range')
                        with tmp.open('ab') as f:
                            shutil.copyfileobj(r, f, 1024 * 1024)
                if tmp.stat().st_size != size:
                    raise ValueError('incomplete range')
                tmp.replace(dst)
                return size
            except Exception as exc:
                print(f'Range {i}: retry {attempt+1} ({type(exc).__name__})', flush=True)
                if attempt == 7:
                    raise RuntimeError(f'range {i} failed') from None
                time.sleep(min(2 ** attempt, 20))
    completed = 0
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for size in ex.map(fetch, ranges):
            completed += size
            print(f'{args.output.name}: {completed}/{args.size} bytes; {time.monotonic()-started:.0f}s', flush=True)
    assembled = args.output.with_name(args.output.name + '.assembling')
    digest = hashlib.new(args.algorithm)
    with assembled.open('wb') as out:
        for i, start, end in ranges:
            with (parts / f'{i:06d}.chunk').open('rb') as src:
                while True:
                    data = src.read(4 * 1024 * 1024)
                    if not data:
                        break
                    digest.update(data)
                    out.write(data)
    if assembled.stat().st_size != args.size or digest.hexdigest() != args.hash:
        raise RuntimeError('final file size or checksum mismatch')
    assembled.replace(args.output)
    shutil.rmtree(parts)
    if prefix.exists():
        prefix.unlink()
    print(f'VERIFIED {args.output.name} {args.algorithm}={digest.hexdigest()}', flush=True)


if __name__ == '__main__':
    main()
