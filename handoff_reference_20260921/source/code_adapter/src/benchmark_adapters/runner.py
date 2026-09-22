import os
import time
import uuid
from pathlib import Path

from .code_tasks import CODE_KINDS, InvalidCodeSubmission, export_code
from .config import LLMConfig
from .environment import DockerEnvironment
from .swe import SWEAdapter
from .tracing import Budget, BudgetExceeded, TraceRecorder, write_json


def build_recorded_llm(config: LLMConfig, recorder: TraceRecorder, budget: Budget):
    """Construct the collector LLM, keeping sampling visible at the transport boundary."""
    from pydantic import SecretStr

    from .sdk_bridge import RecordedLLM

    sampling = {
        name: getattr(config, name)
        for name in ("seed", "top_p")
        if getattr(config, name) is not None
    }
    # vLLM top_k is an integer and allows -1; the SDK field is a nonnegative float.
    extra_body = {
        name: getattr(config, name)
        for name in ("top_k", "presence_penalty", "min_p", "repetition_penalty")
        if getattr(config, name) is not None
    }
    if config.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": config.enable_thinking}
    if extra_body:
        sampling["litellm_extra_body"] = extra_body
    llm = RecordedLLM(
        model=config.model,
        base_url=config.base_url,
        api_key=SecretStr(os.environ.get(config.api_key_env, "dummy")),
        usage_id="actor",
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        timeout=config.timeout,
        num_retries=config.num_retries,
        native_tool_calling=config.native_tool_calling,
        stream=False,
        caching_prompt=False,
        log_completions=False,
        **sampling,
    )
    llm.attach(recorder, budget)
    return llm


def run_task(config, task, attempt_dir, *, environment=None, run_id=None):
    from openhands.sdk import Agent, Conversation

    from .sdk_bridge import Binding, bind, swe_tools, unbind

    is_code = config.dataset.kind in CODE_KINDS
    if is_code and environment is None:
        raise ValueError("Code benchmarks require an explicitly prepared local environment")
    attempt_dir = Path(attempt_dir)
    if attempt_dir.exists() and any(attempt_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing attempt: {attempt_dir}")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(
        task_id=task.task_id,
        dataset_id=task.dataset_id,
        dataset_revision=task.revision,
        split=task.split,
        attempt_id=attempt_dir.name,
        run_id=run_id or uuid.uuid4().hex,
    )
    recorder = TraceRecorder(attempt_dir, identity)
    write_json(attempt_dir / "public_task.json", task.to_dict())
    write_json(attempt_dir / "profile.json", config.to_dict())
    budget = Budget(
        config.runtime.max_tool_calls, config.runtime.max_llm_requests, config.runtime.task_timeout
    )
    env = environment
    if env is None:
        env = (
            DockerEnvironment(config, task, attempt_dir / "artifacts", identity=identity)
            if config.dataset.kind == "swe"
            else None
        )
    result = dict(
        **identity,
        execution_status="environment_error",
        artifact_status="export_failed",
        evaluation_status="pending",
        termination_reason=None,
        trace_quality="incomplete",
    )
    conv = None
    binding_key = None
    prepared = False
    start = time.monotonic()
    recorder.emit("task_start")
    try:
        if config.dataset.kind == "swe" or is_code:
            metadata = env.prepare()
            prepared = True
            for key in ("environment_id", "container_id"):
                if key in metadata:
                    recorder.identity[key] = result[key] = metadata[key]
            recorder.emit("environment_ready", metadata=metadata)
        else:
            from .retrieval import RetrievalEnvironment

            env = RetrievalEnvironment(config)
            metadata = env.prepare()
            prepared = True
            recorder.emit("environment_ready", metadata=metadata)
        result["execution_status"] = "llm_error"
        llm = build_recorded_llm(config.llm, recorder, budget)
        binding_key = bind(
            Binding(
                env, recorder, budget, config.runtime.tool_timeout, config.runtime.max_output_chars
            )
        )
        if config.dataset.kind == "swe" or is_code:
            from .sdk_bridge import code_tools

            tools = code_tools(binding_key) if is_code else swe_tools(binding_key)
            system = (
                "You are an autonomous software engineering assistant. Use only the supplied workspace tools "
                "to inspect and fix the issue. Follow their stateless-shell and file-editing contracts. "
                "Keep all task changes inside the provided repository. When finished, call finish with a summary."
            )
        else:
            from .retrieval import retrieval_tools

            tools = retrieval_tools(binding_key)
            system = (
                "Answer the question using only the fixed corpus tools. Search and read evidence as needed. "
                "Follow the task output format exactly. Call finish with your final answer."
            )
        recorder.environment_tool_names = {tool.name for tool in tools}
        agent = Agent(
            llm=llm,
            tools=tools,
            system_prompt=system,
            condenser=None,
            include_default_tools=["FinishTool", "ThinkTool"],
            tool_concurrency_limit=1,
        )
        workspace = attempt_dir / "conversation_workspace"
        workspace.mkdir()
        conv = Conversation(
            agent=agent,
            workspace=str(workspace),
            callbacks=[recorder.sdk_event],
            max_iteration_per_run=config.runtime.max_iterations,
            persistence_dir=str(attempt_dir / "sdk_state"),
            visualizer=None,
        )
        recorder.identity["conversation_id"] = str(conv.state.id)
        result["conversation_id"] = str(conv.state.id)
        recorder.emit("conversation_start")
        conv.send_message(task.instruction)
        conv.run()
        budget.check()
        state = conv.state.execution_status.value
        result["sdk_execution_status"] = state
        if budget.reason:
            result["execution_status"] = "budget_exhausted"
            result["termination_reason"] = budget.reason
        elif "MaxIterationsReached" in recorder.error_codes:
            result["execution_status"] = "budget_exhausted"
            result["termination_reason"] = "max_iterations"
        elif state == "finished":
            result["execution_status"] = "completed"
            result["termination_reason"] = "finished"
        else:
            result["execution_status"] = "agent_error"
            result["termination_reason"] = state
        result["trace_quality"] = "recorded"
    except (Exception, KeyboardInterrupt) as exc:
        try:
            budget.check()
        except BudgetExceeded:
            pass
        result["error_type"] = type(exc).__name__
        result["termination_reason"] = budget.reason or type(exc).__name__
        if isinstance(exc, KeyboardInterrupt):
            result["execution_status"] = "cancelled"
        elif budget.reason:
            result["execution_status"] = "budget_exhausted"
        recorder.emit(
            "task_error", error_type=type(exc).__name__, reason=result["termination_reason"]
        )
    finally:
        recorder.emit("conversation_end", execution_status=result["execution_status"])
        if prepared:
            try:
                env.quiesce()
                if config.dataset.kind == "swe":
                    patch = env.export_patch()
                    (attempt_dir / "artifacts").mkdir(exist_ok=True)
                    (attempt_dir / "artifacts" / "model.patch").write_text(patch)
                    submission = SWEAdapter.submission(task, patch, config.llm.model)
                    result["artifact_status"] = "valid" if patch else "empty"
                elif is_code:
                    code = export_code(env, task.public_metadata["solution_path"])
                    (attempt_dir / "artifacts").mkdir(exist_ok=True)
                    (attempt_dir / "artifacts" / "solution.py").write_text(code)
                    submission = dict(
                        task_id=task.task_id,
                        dataset_id=task.dataset_id,
                        code=code,
                        model=config.llm.model,
                    )
                    if config.dataset.kind == "livecodebench":
                        submission = dict(question_id=task.task_id, code_list=[code])
                    result["artifact_status"] = "valid" if code.strip() else "empty"
                else:
                    from .retrieval import export_answer

                    submission = export_answer(config.dataset.kind, task, recorder, result, env)
                    result["artifact_status"] = "valid"
                write_json(attempt_dir / "submission.json", submission)
                recorder.emit(
                    "artifact_exported", path="submission.json", status=result["artifact_status"]
                )
            except Exception as exc:
                result["artifact_status"] = (
                    "invalid_submission"
                    if isinstance(exc, InvalidCodeSubmission)
                    else "export_failed"
                )
                result["export_error_type"] = type(exc).__name__
                recorder.emit("artifact_export_error", error_type=type(exc).__name__)
        if conv is not None:
            try:
                conv.close()
            except Exception as exc:
                result["conversation_close_error"] = type(exc).__name__
        if binding_key:
            unbind(binding_key)
        if env is not None:
            try:
                for cleanup_attempt in range(2):
                    try:
                        cleanup = env.close()
                        break
                    except Exception:
                        if cleanup_attempt == 1:
                            raise
                recorder.emit("environment_cleanup", metadata=cleanup, cleanup_status="completed")
            except Exception as exc:
                result["environment_cleanup_error"] = type(exc).__name__
                recorder.emit(
                    "environment_cleanup",
                    metadata=getattr(env, "metadata", None),
                    cleanup_status="failed",
                    error_type=type(exc).__name__,
                )
        result.update(
            duration_s=time.monotonic() - start,
            tool_calls=budget.tools,
            llm_requests=budget.requests,
            final_text=recorder.final_text,
            tool_call_counts=dict(recorder.executed_counts),
            errors=recorder.error_codes,
        )
        write_json(attempt_dir / "result.json", result)
        recorder.emit("task_end", result=result)
        recorder.close()
    return result
