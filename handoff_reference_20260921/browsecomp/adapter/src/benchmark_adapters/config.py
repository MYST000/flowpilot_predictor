import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import get_args

from .contracts import ConfigurationError

SDK_COMMIT = "a6db5dcba26a3acfaeac58c8ba5195433a0e223d"
SDK_PATH = str(Path(__file__).resolve().parents[4])


def _validate_types(cls, values, section):
    annotations = {f.name: f.type for f in fields(cls)}
    for key, value in values.items():
        expected = get_args(annotations[key]) or (annotations[key],)
        if type(value) in expected or (float in expected and type(value) is int):
            continue
        names = " or ".join(kind.__name__ for kind in expected)
        raise ConfigurationError(f"{section}.{key} must be {names}")


@dataclass(frozen=True)
class DatasetConfig:
    kind: str = "swe"
    id: str = "princeton-nlp/SWE-bench"
    path: str = ""
    revision: str = ""
    split: str = "dev"
    sha256: str = ""
    setting: str = "openhands-code-v1"


@dataclass(frozen=True)
class LLMConfig:
    model: str = "openai/qwen3.5-9b"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout: int = 180
    num_retries: int = 2
    native_tool_calling: bool = True
    seed: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    enable_thinking: bool | None = None

    def __post_init__(self):
        _validate_types(type(self), asdict(self), "llm")
        for name in ("temperature", "top_p", "presence_penalty", "min_p", "repetition_penalty"):
            value = getattr(self, name)
            if type(value) is float and not math.isfinite(value):
                raise ConfigurationError(f"llm.{name} must be finite")
        if self.temperature < 0:
            raise ConfigurationError("llm.temperature must be non-negative")
        if self.seed is not None and not 0 <= self.seed < 2**63:
            raise ConfigurationError("llm.seed must be between 0 and 2**63 - 1")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ConfigurationError("llm.top_p must be greater than 0 and at most 1")
        if self.top_k is not None and self.top_k != -1 and self.top_k <= 0:
            raise ConfigurationError("llm.top_k must be -1 (disabled) or a positive integer")
        if self.presence_penalty is not None and not -2 <= self.presence_penalty <= 2:
            raise ConfigurationError("llm.presence_penalty must be between -2 and 2")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ConfigurationError("llm.min_p must be between 0 and 1")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ConfigurationError("llm.repetition_penalty must be positive")


@dataclass(frozen=True)
class RuntimeConfig:
    sdk_path: str = SDK_PATH
    sdk_commit: str = SDK_COMMIT
    max_iterations: int = 60
    max_tool_calls: int = 120
    max_llm_requests: int = 80
    task_timeout: int = 3600
    tool_timeout: int = 120
    max_output_chars: int = 16000
    runs_dir: str = "/root/predictor_exp/runs/benchmark_adapters"


@dataclass(frozen=True)
class DockerConfig:
    image_template: str = "swebench/sweb.eval.x86_64.{image_id}:latest"
    repo_dir: str = "/testbed"
    conda_env: str = "testbed"
    network_disabled: bool = True
    memory_limit: str = "8g"
    nano_cpus: int = 2000000000
    pids_limit: int = 512
    pull_missing: bool = False
    keep_container: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    index_path: str = ""
    corpus_revision: str = ""
    top_k: int = 5
    snippet_chars: int = 1200
    read_chars: int = 6000


@dataclass(frozen=True)
class EvaluationConfig:
    python: str = ""
    harness_path: str = "/root/predictor_exp/repos/SWE-bench"
    workers: int = 1
    timeout: int = 1800
    namespace: str = "swebench"
    image_tag: str = "latest"
    hotpot_script: str = ""
    browsecomp_repo: str = "/root/predictor_exp/repos/BrowseComp-Plus"
    judge_model: str = "Qwen/Qwen3-32B"


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self):
        data = asdict(self)
        # Unset sampling controls retain legacy profile hashes and provider defaults.
        data["llm"] = {key: value for key, value in data["llm"].items() if value is not None}
        return data

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    classes = dict(
        dataset=DatasetConfig,
        llm=LLMConfig,
        runtime=RuntimeConfig,
        docker=DockerConfig,
        retrieval=RetrievalConfig,
        evaluation=EvaluationConfig,
    )
    if unknown := raw.keys() - classes.keys():
        raise ConfigurationError(f"Unknown config sections: {sorted(unknown)}")
    objects = {}
    for section, cls in classes.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ConfigurationError(f"{section} must be a TOML table")
        if unknown := values.keys() - {f.name for f in fields(cls)}:
            raise ConfigurationError(f"Unknown {section} keys: {sorted(unknown)}")
        _validate_types(cls, values, section)
        host_paths = {
            "dataset": {"path"},
            "runtime": {"sdk_path", "runs_dir"},
            "retrieval": {"index_path"},
            "evaluation": {"python", "harness_path", "hotpot_script", "browsecomp_repo"},
        }
        for key in list(values):
            if values[key] and key in host_paths.get(section, set()):
                host_path = path.parent / values[key]
                # Resolving a venv Python symlink bypasses pyvenv.cfg and loses its packages.
                values[key] = str(host_path.absolute() if key == "python" else host_path.resolve())
        objects[section] = cls(**values)
    cfg = Config(**objects)
    if cfg.dataset.kind not in {
        "swe",
        "hotpot",
        "browsecomp",
        "quixbugs",
        "livecodebench",
        "classeval",
    }:
        raise ConfigurationError("Unsupported dataset.kind")
    if cfg.dataset.id.endswith("SWE-bench_Verified") and cfg.dataset.split != "test":
        raise ConfigurationError("SWE-bench Verified only has the test split")
    supported = {
        "swe": {"princeton-nlp/SWE-bench", "princeton-nlp/SWE-bench_Verified"},
        "hotpot": {"hotpotqa"},
        "browsecomp": {"Tevatron/browsecomp-plus"},
        "quixbugs": {"jkoppel/QuixBugs"},
        "livecodebench": {"livecodebench/code_generation_lite"},
        "classeval": {"FudanSELab/ClassEval"},
    }
    if cfg.dataset.id not in supported[cfg.dataset.kind]:
        raise ConfigurationError("Unsupported dataset identity for selected adapter")
    settings = {
        "swe": "openhands-code-v1",
        "hotpot": "fullwiki-fixed-corpus-v1",
        "browsecomp": "openhands-search-read-v1",
        "quixbugs": "openhands-python-code-v1",
        "livecodebench": "openhands-python-code-v1",
        "classeval": "openhands-python-code-v1",
    }
    if cfg.dataset.setting != settings[cfg.dataset.kind]:
        raise ConfigurationError(
            f"Unsupported dataset.setting; expected {settings[cfg.dataset.kind]}"
        )
    allowed_splits = {"train", "dev", "test"}
    if cfg.dataset.kind == "livecodebench":
        allowed_splits |= {"fit", "tune", "calibration", "historical_dev"}
    if cfg.dataset.split not in allowed_splits:
        raise ConfigurationError("Unknown dataset split")
    if cfg.dataset.kind == "browsecomp" and cfg.dataset.split != "test":
        raise ConfigurationError("BrowseComp-Plus question data only has test split")
    if not cfg.dataset.path or not cfg.dataset.revision:
        raise ConfigurationError("dataset.path and dataset.revision are required")
    for section in (cfg.runtime, cfg.retrieval, cfg.evaluation, cfg.docker):
        for f in fields(section):
            value = getattr(section, f.name)
            if type(value) is int and value <= 0:
                raise ConfigurationError(f"{f.name} must be positive")
    if cfg.llm.max_output_tokens <= 0 or cfg.llm.timeout <= 0 or cfg.llm.num_retries < 0:
        raise ConfigurationError("Invalid LLM token/timeout/retry budget")
    if not cfg.llm.native_tool_calling:
        raise ConfigurationError("Prediction tracing requires llm.native_tool_calling=true")
    if cfg.runtime.sdk_commit != SDK_COMMIT:
        raise ConfigurationError(f"This implementation supports the pinned SDK {SDK_COMMIT}")
    if not cfg.docker.repo_dir.startswith("/") or cfg.docker.repo_dir == "/":
        raise ConfigurationError("docker.repo_dir must be an absolute task directory")
    return cfg
