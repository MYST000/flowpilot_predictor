"""Check the pinned wheel without installing or importing vLLM."""
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
wheel = ROOT/'downloads/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
h = hashlib.sha256()
with wheel.open('rb') as f:
    for data in iter(lambda:f.read(8*1024*1024),b''):
        h.update(data)
assert wheel.stat().st_size == 548493389
assert h.hexdigest() == '22e8d8fec755986b3ad964004a1f8c65a55626ec948354bdbe95993e6b0289fe'
with zipfile.ZipFile(wheel) as z:
    assert z.testzip() is None
    names = [n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
    assert len(names)==1
    metadata = Parser().parsestr(z.read(names[0]).decode())
    assert metadata['Name']=='vllm' and metadata['Version']=='0.29.0+cu129'
    (ROOT/'evidence/preparation/vllm_wheel_METADATA.txt').write_text(z.read(names[0]).decode())
result = {'path': str(wheel.relative_to(ROOT)), 'source': 'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl', 'additional_official_download_source': 'https://wheels.vllm.ai/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl', 'bytes': wheel.stat().st_size, 'sha256': h.hexdigest(), 'version': metadata['Version'], 'requires_python': metadata['Requires-Python'], 'requires_dist': metadata.get_all('Requires-Dist', []), 'sha256_verified_against_official_release': True, 'zip_crc_check': 'passed', 'installed': False, 'service_started': False}
(ROOT/'evidence/preparation/vllm_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
print('VLLM_WHEEL_VERIFIED', result['version'], 'Python',result['requires_python'])
print('Core requirements:',[r for r in result['requires_dist'] if r.startswith(('torch', 'transformers', 'tokenizers', 'flashinfer'))])
