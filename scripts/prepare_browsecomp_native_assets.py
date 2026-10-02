"""Download fixed official BM25/tokenizer/Java assets, verifying upstream digests."""
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/retrieval'
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def download(job):
    url, dst, size, algorithm, expected = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    if size > 100_000_000 and algorithm == 'sha256' and not dst.exists():
        subprocess.run([
            '/usr/bin/python3', str(ROOT / 'scripts/download_ranges.py'),
            '--url', url, '--output', str(dst), '--size', str(size),
            '--hash', expected, '--workers', '8', '--resolve-via-proxy',
        ], check=True)
    elif not dst.exists():
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
            resolved = response.url
        opener = urllib.request.build_opener() if urllib.parse.urlparse(resolved).hostname in {'huggingface.co', 'hf.co'} else DIRECT
        with opener.open(resolved, timeout=120) as response, dst.with_suffix(dst.suffix+'.part').open('wb') as stream:
            while chunk := response.read(1024*1024):
                stream.write(chunk)
        dst.with_suffix(dst.suffix+'.part').replace(dst)
    raw_hash=hashlib.sha256()
    git_hash=hashlib.sha1(f'blob {size}\0'.encode())
    with dst.open('rb') as stream:
        while chunk := stream.read(4*1024*1024):
            raw_hash.update(chunk);git_hash.update(chunk)
    actual=raw_hash.hexdigest() if algorithm=='sha256' else git_hash.hexdigest()
    if dst.stat().st_size != size or actual != expected:
        raise ValueError(f'Official checksum mismatch: {dst.name}')
    print(f'VERIFIED {dst.name} {size}', flush=True)
    return {'path':str(dst),'url':url,'bytes':size,'sha256':raw_hash.hexdigest(),
            'upstream_algorithm':algorithm,'upstream_hash':expected}

jobs=[]
for kind, repo, lock, prefix, names in [
    ('datasets','Tevatron/browsecomp-plus-indexes','bc_indexes_lock','data/browsecomp_plus/native_indexes',None),
    ('models','Qwen/Qwen3-0.6B','tokenizer_lock','runtime/browsecomp-native/tokenizer',
     {'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','config.json','LICENSE'}),
]:
    info=json.loads((EVIDENCE/f'flowpilot_{lock}.json').read_text())
    for f in info['siblings']:
        name=f['rfilename']
        if not ((name.startswith('bm25/') if names is None else name in names)):
            continue
        url=f'https://huggingface.co/{"datasets/" if kind=="datasets" else ""}{repo}/resolve/{info["sha"]}/{name}'
        jobs.append((url,ROOT/prefix/name,f['size'],'sha256' if 'lfs' in f else 'git-sha1',
                     f['lfs']['sha256'] if 'lfs' in f else f['blobId']))
java=json.loads((EVIDENCE/'flowpilot_java21_info.json').read_text())[0]['binary']['package']
jobs.append((java['link'],ROOT/'downloads'/java['name'],java['size'],'sha256',java['checksum']))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results=list(executor.map(download,jobs))
(EVIDENCE/'verified_native_assets.json').write_text(json.dumps(results,indent=2)+'\n')
