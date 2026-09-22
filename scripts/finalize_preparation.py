"""Write a completion receipt only after every requested asset has passed validation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'evidence/preparation'
def read(name):
    return json.loads((EVIDENCE/name).read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):
            h.update(data)
    return h.hexdigest()

assets=read('dataset_preparation.json')
assets['hotpot_corpus']=read('hotpot_corpus_preparation.json')
assets['vllm']=read('vllm_preparation.json')
assert assets['vllm']['zip_crc_check']=='passed'
assert assets['hotpot_corpus']['archive_and_shard_integrity']=='passed'
originals={'交接文档.md':'bda6180999a6a8403864cc610fce9a4327545452d23aedf56f681b910066d5dd','交接（包含源码与校验）.md':'d71067fa7bb9ad7506ea2a31d11e3339a21fb55e3e780bc7147df5694d77b6fd'}
for name,expected in originals.items():
    assert sha(ROOT/name)==expected, 'Original handoff modified'
space=shutil.disk_usage(ROOT)
receipt={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'project_root':str(ROOT),'status':'all_requested_downloads_verified','assets':assets,'hf_source_lock':'evidence/preparation/hf_sources.lock.json','hf_file_checksums':'evidence/preparation/hf_verified_files.json','original_handoffs_sha256':originals,'data_environment':{'python':sys.version.split()[0],'path':'.venv-data','required_packages':'scripts/requirements-data.txt'},'disk':{'free_bytes':space.free,'total_bytes':space.total},'performed': ['download pinned repositories and datasets','verify source hashes, archive integrity, task counts and document references','restore frozen LCB selection and split files','decode BrowseComp query and relevance fields locally','extract Hotpot compressed Wikipedia shards'],'not_performed':['model weights download','OpenHands or vLLM service environment installation','retrieval index construction','adapter concurrency changes','model startup','data collection','training or benchmark experiments']}
(EVIDENCE/'preparation_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
entries=read('hf_verified_files.json')
for key in ['lcb','hotpot_corpus','vllm']:
    asset=assets[key]
    path=asset.get('path',asset.get('archive'))
    entries.append({'path':path,'sha256':asset['sha256']})
entries.append({'path':assets['browsecomp']['prepared_queries'],'sha256':assets['browsecomp']['prepared_queries_sha256']})
for path in sorted((ROOT/'scripts').glob('*')):
    if path.is_file(): entries.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
for path in sorted((ROOT/'data/livecodebench/protocol').glob('*')):
    if path.is_file():entries.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
for rel in ['data/livecodebench/testing_util.reference.py','data/browsecomp_plus/browsecomp_known_history.ids']+list(originals):
    entries.append({'path':rel,'sha256':sha(ROOT/rel)})
(EVIDENCE/'SHA256SUMS').write_text(''.join(f"{e['sha256']}  {e['path']}\n" for e in entries))
(ROOT/'downloads/SHA256SUMS').write_text(f"{assets['vllm']['sha256']}  {Path(assets['vllm']['path']).name}\n")
free_tib=space.free/1024**4
text=f'''# 实验准备下载清单

本次要求下载的源码、数据和 vLLM wheel 已完成下载与校验。主目录：`{ROOT}`。

| 项目 | 已固定版本与规模 | 本地位置 |
|---|---|---|
| OpenHands 用户 fork | `ql`，提交 `8e8ed520c8abb09c4c4d3369492d3b02988e2c18`，工作树完整、Git 校验通过 | `repos/Openhands-software-agent-sdk/` |
| HotpotQA fullwiki 问题 | train 90,447；validation 7,405；test 7,405 | `data/hotpotqa/questions/fullwiki/` |
| HotpotQA 固定 Wikipedia 语料 | 官方 2017-10-01 fullwiki 简介段落语料；{assets['hotpot_corpus']['documents']:,} 条文档，{assets['hotpot_corpus']['compressed_shards']:,} 个压缩分片 | 原包 `data/hotpotqa/raw/`；解包 `data/hotpotqa/corpus/` |
| BrowseComp-Plus | 830 题；100,195 篇固定文档；全部标注文档 ID 已核对存在 | `data/browsecomp_plus/questions/`、`data/browsecomp_plus/corpus/` |
| LiveCodeBench | 固定 release_v6，原始 1,055 题；依交接清单恢复 362 题 | `data/livecodebench/release_v6/`、`data/livecodebench/lcb362.jsonl` |
| QuixBugs（你所说的 Quick 40） | 提交 `4257f44b0ff1181dedaedee6a447e133219fcebf`；40 题的错误代码、修复参考和测试均存在 | `data/QuixBugs/` |
| vLLM | `0.29.0+cu129`，CUDA 12.9 / Linux x86_64 wheel | `downloads/{Path(assets['vllm']['path']).name}` |

## 版本与数据处理

- Hotpot 问题文件使用 `hotpotqa/hotpot_qa` 的固定提交 `1908d6afbbead072334abe2965f91bd2709910ab`，下载 fullwiki Parquet 格式。旧 CMU 问题文件 HTTPS 地址不可用，因此记录了所用 HF 来源；没有把问题中的 context 当成检索语料库。
- Hotpot Wikipedia 使用官方 Stanford `enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`，包含官方 fullwiki 使用的简介段落。官方 MD5 `01edf64cd120ecc03a2745352779514c` 已匹配，外层归档与全部内部 BZip2/JSONL 分片已读取验证。检索应使用 `text` 字段；本次未建索引。
- BrowseComp-Plus 的原始题目和相关性标注 Parquet 原样保留。官方解码函数生成 `data/browsecomp_plus/queries_private.jsonl`，包含 query、answer 和 gold/evidence/negative 文档 ID；文档正文保留在固定语料和原始 Parquet 中。该文件属于控制器私有数据，不应整体送给 actor。历史已接触 ID 清单也已保留，没有重新划分题目。
- LCB 362 题逐题核对题干哈希及公开接口；划分仍是 historical_dev **25**、fit **205**、tune **51**、calibration **31**、test **50**。清单、ID 文件、历史接触记录均在 `data/livecodebench/protocol/`；上游判题器固定版本为 `data/livecodebench/testing_util.reference.py`。隐藏测试原样保留，未执行其解包或判题代码。
- QuixBugs 40 题清单是 `data/livecodebench/protocol/quixbugs_dev40.ids`；按交接协议用于开发/回归，不构成新的未见测试集。
- OpenHands 保持用户指定 `ql` 分支；适配器位于 `benchmarks/flowpilot/`。本次采用浅克隆，未改变代码或串行执行方式。交接文档中的 142 个嵌入文件另存于 `handoff_reference_20260921/`，两个原始 Markdown 文件未改动。

## 校验记录

- 完整下载与处理清单：`evidence/preparation/preparation_manifest.json`。
- HF 固定提交、文件大小与官方哈希：`evidence/preparation/hf_sources.lock.json`。
- 本地文件 SHA256：`evidence/preparation/SHA256SUMS`（路径相对主目录）。
- vLLM SHA256：`{assets['vllm']['sha256']}`，与官方 GitHub Release 一致；ZIP CRC 校验通过。
- vLLM 实际依赖元数据：`evidence/preparation/vllm_wheel_METADATA.txt`；wheel 要求 Python `{assets['vllm']['requires_python']}`。
- 大文件优先直连；vLLM 同时使用 GitHub Release 与官方 `wheels.vllm.ai` 的相同文件续传，完整 SHA256 一致。未修改本机全局代理设置。
- 本地校验/准备脚本在 `scripts/`；数据读取环境为 `.venv-data`，所需包为 `pyarrow==25.0.1`。这不是 OpenHands 或 vLLM 服务环境。

## 当前边界与下一步

当前文件系统剩余约 **{free_tib:.2f} TiB**，本轮下载及后续数据存储有充足磁盘空间。

此次完成下载和确定性数据准备，**尚未安装 OpenHands/vLLM 服务环境、下载 Qwen 权重、构建检索索引、改并发适配器或开展任何采集/训练实验**。wheel 校验通过不代表模型推理兼容性已实测。

下一阶段可按交接文档配置独立 Python 3.12 服务/控制器环境，安装固定依赖，然后实现两个固定语料库的检索 runtime 与统一轨迹字段；并发适配器和小规模联调放在这些准备完成之后。
'''
(ROOT/'准备完成清单_2026-09-21.md').write_text(text)
print('PREPARATION_COMPLETE', f'disk_free_TiB={free_tib:.2f}')
