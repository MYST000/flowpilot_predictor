# GitHub 发布与共享 27B LightGBM 使用手册

本手册适用于 2026-10-02 的预测侧交付。模型留在共享 `/data`，GitHub 只发布代码、文档、算法参数/依赖锁定文件和适配器源码。

## 1. 已有模型与必须保留的文件

| 项目 | 路径 |
| --- | --- |
| 正式 raw LightGBM | `/data/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1/train/lightgbm/model.joblib` |
| 模型 manifest | `/data/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1/train/lightgbm/manifest.json` |
| 准备数据 metadata | `/data/ql_flowpilot_predictor/predictor_prepared/native27b_1077_v1/manifest.json` |
| 训练时的 predictor 代码快照 | `/data/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1/deployment_snapshot/code/predictor/` |

模型版本：`tool-rtt-v2:lightgbm:a54e331d027f9348`。

模型文件 SHA256：

```text
0432b075b8db58068c70326cddd84ec8aa02084a5ad0f9db4d00fb9b4a75db13
```

模型训练集 9,335 条工具 RTT（包含仅用于辅助训练的 QuixBugs），四个 LightGBM 分位头为 Q10/Q50/Q90/Q99。正式桥接加载 raw 工件，再使用在线校正，不加载离线静态校准工件。

当前模型目录权限为 `700`，模型及其 manifest 为 `600`；若同门使用另一 Linux 用户，仅能访问 `/data` 顶层还不够，需要额外只读授权。使用同一 root 账号读取则无需改权限。发布操作不自动放宽共享文件权限。

共享访问必须同时满足：同门所在机器挂载的是**同一套文件内容**，路径一致，且用户可遍历所有父目录并读取上述文件。另一台机器有同名 `/data` 不代表共享成功。若挂载点不同，只修改自己本地 `configs/predictor/runtime.json` 的三个路径；不修改模型/manifest，也不取消校验。

由共享目录管理者针对同门实际 Linux 用户授权后，在同门账号执行以下只读检查：

```bash
namei -l /data/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1/train/lightgbm/model.joblib
test -r /data/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1/train/lightgbm/model.joblib && echo "model readable"
```

需要授权的是上述文件的读取和所有父目录的遍历，不需要共享目录写权限，也不需要递归授权所有实验数据。

原工件是完整训练序列化对象，内部保留训练样本用于原算法的支持度核验，因此它继续只保存在共享磁盘，本次不上传。加载只需上述 metadata/代码，不需读取训练 JSONL。不要删改这些配套文件。

## 2. 上传者：准备独立源码目录

本机已准备的目录为：

```text
/root/flowpilot_predictor_github_20261002
```

它不含 `.git`，不改变原 `/root/flowpilot_predictor` 的历史或暂存区。模型、数据、语料索引、虚拟环境、缓存、实验日志、实际服务器配置和密钥不包含在内；依赖版本文件、通用算法参数和共享模型路径模板保留。历史源码/文档中的路径作为实验溯源文字保留，不意味着发布相应环境文件。

以后重新导出时使用一个尚不存在的输出目录：

```bash
/root/flowpilot_predictor/.venv-predictor/bin/python -B \
  /root/flowpilot_predictor/scripts/prepare_github_source.py \
  --output /root/flowpilot_predictor_github_NEXT
```

脚本保存两个版本的 OpenHands benchmark 适配器源码，包括尚未提交的修改和新增文件，避免只上传子模块指针而丢失修改。它不推送 GitHub，不删除原文件，也不导出模型。

## 3. 上传者：首次推送

在准备目录运行下列命令。原仓库暂存区已有历史日志和配置，因此本次不要在原目录直接执行 `git add .`。

```bash
cd /root/flowpilot_predictor_github_20261002
git init -b main
git remote add origin https://github.com/Fate0123/flowpilot_predictor.git
git config user.name "Fate0123"
git config user.email "Fate0123@users.noreply.github.com"
git fetch origin
```

`user.email` 可改成 GitHub 设置页面显示的准确 noreply 邮箱或已验证邮箱。HTTPS 提示身份时，用户名填 `Fate0123`，密码位置填有目标仓库写权限的 GitHub token。不要把 token 写入远端 URL、手册或命令参数。本机读取远端 refs 时返回需要认证，因此推送前需要完成 GitHub 身份认证。

如果远端已有 `main`，接上它的历史，但保持当前干净导出文件不变：

```bash
if git show-ref --verify --quiet refs/remotes/origin/main; then
  git reset --mixed origin/main
fi
git add -A
git diff --cached --stat
git status --short
```

这只在独立上传目录操作：已有远端中的日志/本机配置会从最新版本移除，原服务器文件不变，已有远端提交历史也不被重写。如远端采用其他主分支，请把上面的 `main` 换成实际主分支名。确认暂存清单后：

```bash
git commit -m "Publish 27B RTT predictor, realtime calibration and shared-model guide"
git push -u origin main
```

不使用 force push。若他人在 fetch 后更新远端而导致推送拒绝，先 `git fetch origin`，审查/处理新提交后再推送。尚未配置 GitHub token 时，完成本地 commit 后即可等待认证，不影响现有实验或共享模型。

已有远端历史中的旧日志/配置不会被这次普通提交从历史中清除；本次只保证新版本源码目录不包含它们。本次不执行历史重写。

## 4. 同门：安装与验证

```bash
git clone https://github.com/Fate0123/flowpilot_predictor.git
cd flowpilot_predictor
conda create -n flowpilot-predictor-27b python=3.12.9 pip -y
conda activate flowpilot-predictor-27b
python -m pip install -r scripts/requirements-predictor-lock.txt
python -m pip check
cp configs/predictor/shared-data1.example.json configs/predictor/runtime.json
python -B scripts/check_shared_model.py
```

依赖严格锁定：Python 3.12.9、NumPy 2.2.6、SciPy 1.15.3、scikit-learn 1.6.1、LightGBM 4.6.0、joblib 1.4.2、threadpoolctl 3.6.0。不能只安装“Python 3.12 最新版”或升级 LightGBM；当前加载器会核验精确版本。Linux 需要可用的 OpenMP 动态库；若提示 `libgomp.so.1` 缺失，在相应系统安装 `libgomp1`。

检查成功显示 `status: ready`、四个预测分位数和 `workers: 4`。不会执行工具、启动 Qwen/vLLM/采集任务或写共享数据。独立检查没有调度 sink，显示 `duration_sink_bound: false` 正常；不能据此声称完整框架已运行。

常见启动失败：

- `PermissionError`：由共享磁盘管理者按用户组授权读取模型及父目录；不要对整个 `/data` 递归放宽权限。
- `model/runtime dependency mismatch`：按精确 Python/依赖锁文件建立独立环境。
- `saved model/code snapshot mismatch` / `runtime model implementation mismatch`：使用本次交付的预测代码和模型对应快照，检查 SHA；不要绕过校验。
- `data manifest mismatch`：`prepared_data` 指向 `native27b_1077_v1`，不要指向旧 9B 数据准备目录。
- `unknown_backend_tool_or_version`：请求的后端/语料版本/schema 与训练域不一致。使用当前适配器提供的真实身份，必要时重新收集/训练，不伪造版本使它通过。

## 5. 默认实时预测行为

共享模板保持当前实际配置：四个预测副本、`selected_quantile: q50`、`online: true`、`online_method: quantile_residual_v1`；不启动预测服务以外的任何模型进程。

在线参数：alpha=0.1、shrinkage_rows=256、window=1,024、每次 log-offset 最大变化 0.02、整体 log-offset 限幅 ±log(2)。Q10/Q50 启用校正至少需 32 调用/4 任务，Q90 为 100/10，Q99 为 1,000/30。少量样本时仍记录反馈但该头输出基础模型值。不同 tool/backend/version/schema 独立维护校正状态；这些阈值不代表覆盖率保证。

`workers: 4` 是 CPU 预测副本数，和 Qwen 的四卡 TP、benchmark 四任务并发是独立配置。本包只负责预测；原 LLM/任务并发及 KV/cache/调度策略沿用同门的配置。

LightGBM 树权重保持冻结，实时更新的是逐分位残差校正层。在线状态在内存，进程重启冷启动；重启不会修改共享模型。

## 6. 输入、输出和真实反馈

`PredictorRuntime.submit(ToolPredictionRequest(...))` 在 LLM 完整工具调用解析完成（`stage=tool_call_ready`）后立即执行；命中查找同时执行。每个请求需要完整 `CallIdentity`，包含 job/line/request/tail/LLM/tool-call ID 与 attempt/epoch/version。

预测上下文必需字段：`backend_id`、`backend_version`、`tool_name`、`tool_schema_version`、完整解析并补默认值的 `arguments`、`execution_mode`。历史工具客户端 RTT、T0 负载、采样年龄、timeout/batch/预算字段按现有适配器提供。不得输入本次工具返回结果或实际耗时等未来信息。

输出：

```text
duration_ms = {q10, q50, q90, q99}  # 单位 ms
duration_estimate_ms              # 当前选中的 q50/q90
selected_quantile
predictor_version / prediction_id / online_state_version
support / fallback / online_calibration
```

正式输出进入已有 `ToolResolutionStore` 的 `duration_estimate_ms` 和已有 refresh 回调。调度排队/offload/reload 的决策仍由同门算法处理，本包不修改策略。需要切换 Q90 时只改本地配置 `selected_quantile`；调度接口当前只接收 Q50/Q90，Q10/Q99 用于记录与分析。

只有真实本地执行（`LOCAL_ONLY`/`LOCAL_LEADER`）完成或执行错误返回的客户端 dispatch→return `round_trip_ms` 才调用 `feedback` 更新。缓存 hit、inflight follower、cancelled/blocked 不作为本地执行 RTT。调用身份与 event ID 去重；当前反馈只影响之后提交的预测，不能泄漏到当前预测。

## 7. 接到同门的完整 FlowPilot

先安装同门的 FlowPilot（使用 `DEPENDENCIES.json` 记录的兼容提交；确认已包含 duration adapter 注册接口），在同一个精确 ML 环境中执行：

```bash
# 替换成同门现有仓库目录。
python -m pip install -c scripts/requirements-predictor-lock.txt -e /path/to/flowpilot
export FLOWPILOT_PREDICTOR_CONFIG="$PWD/configs/predictor/runtime.json"
export PYTHONPATH="$PWD:/path/to/flowpilot${PYTHONPATH:+:$PYTHONPATH}"
```

同门已有 API key、DCS key、实例、复用注册表、引擎成本表和实验 profile 沿用原配置，不从本仓库导出本机凭据。注册表应保持本机原四类复用工具：`hotpot/search`、`hotpot/read_document`、`browsecomp/search`、`browsecomp/get_document`；预测对其他受支持工具也执行，不仅对可复用工具执行。

已有 profile 入口可通过包装器接入（变量由同门原实验设置提供）：

```bash
python -B -m flowpilot_predictor_bridge.launch_gateway \
  --config "$EXPERIMENT_PROFILE" \
  --run-dir "$RUN_DIR" \
  --registry "$TOOL_REGISTRY" \
  --check
```

这个 `--check` 只核验 profile 和模型，不启动网关，也不等于完整实验验收。成本模型路径沿用 `FLOWPILOT_COST_MODEL_PATH` 或已有 `--cost-model`。确定沿用的服务配置正确后去掉 `--check` 才正式启动网关。需要后台持久运行时可在同门原 tmux 启动脚本中**只替换网关入口**，保持原进程管理、GPU、实例和所有策略配置。

已有自定义 app 启动器可直接改为：

```python
from flowpilot_predictor_bridge.serve import create_app
app = create_app(settings)  # 同门原有 Settings 对象
```

它在 app 生命周期中注册 `FrameworkPredictor`、绑定真实调度 sink 和工具反馈 hook。不要直接调用不带 `tool_duration_adapter` 的原 `flowpilot.app.create_app`，否则预测没有开启。框架必须使用真实 RTT，并关闭 synthetic duration。

## 8. OpenHands 适配器交付

`adapters/openhands_current/benchmarks/flowpilot` 是 `/root/flowpilot_integration_20260928/openhands` 中当前使用的源码快照，包含 `predictor_integration.py`、耗时采集与四类查询工具，保留本机未提交和新增修改。`openhands_legacy` 留存早期版本，正式接入使用 current。

这些是 benchmark 扩展，不是整个 SDK 安装包。完整 SDK 从 `DEPENDENCIES.json` 所列仓库/提交获取；合并该目录到对应 SDK 工作副本后按 SDK 原方法安装。也可在已安装兼容 SDK 的环境中安装扩展：

```bash
python -m pip install -e adapters/openhands_current/benchmarks/flowpilot
```

`tracked-changes.patch` 仅包含已跟踪文件相对所列 base commit 的修改；新增文件包含在源码快照中。不能只应用 patch 而漏掉新增文件，也不要同时叠加 legacy/current 两份扩展。已有 SDK 改动先对比后合并，不覆盖同门的其他修改。适配器安装仍需 SDK/容器/检索服务等原有依赖；它不提供语料与 GPU 环境。

### 身份对齐与实验映射

OpenHands 是会话身份的来源。适配器不再设置 `job_id = run_id`，而是沿用 `LocalConversation` 的持久化 UUID：根会话默认生成 `job-<conversation_id>` 和 `line-<conversation_id>`。FlowPilot 先接受 Job/Line 注册，再验证请求头；预测器使用这些已验证的身份，不另行生成 Job 或会话 ID。

| 字段 | 对应关系 |
| --- | --- |
| `run_id` | 一批实验，可关联多个独立任务的 Job，不是调度 Job |
| `run_id/task_id/attempt_id` | 一次 benchmark 任务尝试；通过日志关联该次根会话与 Job |
| `job_id` | 一个工作流；根会话及其子代理共享，独立任务各有自己的 Job |
| `conversation_id/line_id` | 一个 OpenHands 会话及其调度线路；子代理各自持有新的会话和线路 |
| `request_id/attempt/llm_call_id` | 一次逻辑请求及其传输尝试；传输重试保留逻辑请求 ID，递增 attempt 并产生新的 LLM 调用 ID |

恢复同一个持久化会话时沿用其身份；重新创建会话的任务重跑会生成新的 Job。benchmark 的 `attempt_id` 与请求头中的整数 `attempt` 是两个不同概念。

任务目录中的 `events.jsonl` 新增 `flowpilot_request_identity`，顶层保留 `run_id/task_id/attempt_id`，`flowpilot_identity` 保存来自 SDK 请求头的 Job、Line、conversation 和调用身份。日志的顶层 `request_id` 是采集请求 ID，内层 `flowpilot_identity.request_id` 才是 FlowPilot 逻辑请求 ID。DCS 续接还记录 `flowpilot_response_identity`，将最终调用关联到同一采集请求，真实工具 RTT 反馈使用该最终调用身份。

因此，两边通过现有 `X-FlowPilot-*` 请求头、回复中的 `flowpilot.final_identity` 和 `/flowpilot/v1/predictor/feedback` 对齐，无需额外身份分配或映射 RPC。实验 ID 不进入 provider messages，也不参与 Job 公平性计算。

### 预测完成与 DCS

`FrameworkPredictor.on_response()` 同步提交工作，返回本次工具批次的后台 awaitable。完成表示适用的本地预测已经过实际 resolution 和版本校验写入；仅原生模型计算结束还不算交付完成。KV 保留决策收集该信号，正常回复和 OpenHands 工具执行不等待它。预测不可用、超时或写入异常会使该批次预测明确失效；命中、跟随者及已结束的工具无需等待本地耗时。取消收集不会停止原生计算或丢弃合法的延迟 RTT 学习。

DCS 内部请求保留原始预测上下文，并以最外层请求到达网关的时间为统一起点累计快照年龄，包含中间推理与缓存处理时间，不比较不同主机的单调时钟。每轮仍使用自己的请求／LLM 调用身份；私有预测头不会转发到 vLLM。

### 本次修复的本地验证（2026-10-03）

- FlowPilot 全套测试与预测器核心回归：520 passed、3 skipped。跳过项依赖旧工件／历史准备数据；正式 27B 工件加载和快照篡改拒绝测试通过。
- 实际 OpenHands SDK、身份关联及本地 DCS 联通组合：53 passed。交付快照的同一组身份测试另行运行，3 passed。
- FlowPilot 生产代码及本次定向检查的 Ruff/Pyright、SDK 修改文件的 pre-commit、11 个修改模块的编译检查通过。扩大到所有旧测试文件的 Pyright 仍有 20 处既有类型错误，均位于本次未改动的测试文件。
- 联通使用真实 OpenHands、FlowPilot 与本地模拟推理服务；尚未取得真实 vLLM/GPU 调度收益或恢复成本的生产证据。

本机复现命令（其他机器替换仓库和已锁定的模型环境路径）：

```bash
fp_repo=/home/liyachen/workspace/flowpilot
predictor_repo=/home/liyachen/workspace/flowpilot_predictor
sdk_repo=/home/liyachen/openhands/software-agent-sdk
ml_site=/home/liyachen/.conda/envs/flowpilot-predictor-27b/lib/python3.12/site-packages

cd "$fp_repo"
FLOWPILOT_SDK_REPO="$predictor_repo/adapters/openhands_current" \
PYTHONPATH="$ml_site:$predictor_repo:$fp_repo" \
.venv/bin/python -B -m pytest tests \
  "$predictor_repo/tests/test_prediction_completion.py" \
  "$predictor_repo/tests/test_predictor_bridge.py" \
  "$predictor_repo/tests/test_online_quantile_calibration.py" \
  "$predictor_repo/tests/test_time_predictor.py" -q --disable-warnings --tb=short

cd "$sdk_repo"
OPENHANDS_SUPPRESS_BANNER=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
NO_PROXY=127.0.0.1,localhost \
PYTHONPATH="$fp_repo:$sdk_repo/benchmarks/flowpilot/src" \
.venv/bin/python -B -m pytest \
  "$fp_repo/integration/test_predictor_identity.py" \
  "$fp_repo/integration/test_openhands_reuse.py::test_deferred_gateway_preserves_prediction_context_across_rounds" \
  "$fp_repo/integration/test_openhands_reuse.py::test_deferred_gateway_records_inner_tool_hits" \
  tests/sdk/test_flowpilot.py tests/sdk/test_flowpilot_mcp_arguments.py \
  -q --disable-warnings --tb=short

OPENHANDS_SUPPRESS_BANNER=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
PYTHONPATH="$fp_repo:$predictor_repo/adapters/openhands_current/benchmarks/flowpilot/src" \
.venv/bin/python -B -m pytest "$fp_repo/integration/test_predictor_identity.py" \
  -q --disable-warnings --tb=short
```

历史训练/采集脚本需要原数据目录才能复现实验；同门仅调用现成预测器无需运行这些脚本。模型加载、四并发检查成功后，还须在正式框架实测 `duration_estimate_ms` 回写和真实 feedback 更新，才能确认整个系统已接入。

2026-10-04 已完成 BrowseComp-Plus 五题各两次的实机测试前置准备，包含远端 MCP、冻结题目、历史耗时 × 1.5 的 SLO 入口及完整链路启动脚本。因 GPU 资源占用，10 次任务尚未执行；状态、证据与下次命令见 [5 × 2 实机测试交接](BROWSECOMP_5X2_HANDOFF_20261004.md)。
