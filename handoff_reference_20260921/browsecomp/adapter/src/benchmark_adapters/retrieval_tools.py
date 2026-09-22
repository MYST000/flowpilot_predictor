import json
import time

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from pydantic import Field

from .sdk_bridge import get_binding
from .tracing import BudgetExceeded


class SearchAction(Action):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class HotpotReadAction(Action):
    doc_id: str
    start_sentence: int = Field(default=0, ge=0)
    max_sentences: int = Field(default=20, ge=1, le=100)


class BrowseReadAction(Action):
    docid: str
    offset: int = Field(default=0, ge=0)


class RetrievalObservation(Observation):
    pass


class RetrievalExecutor(ToolExecutor):
    def __init__(self, binding, name):
        self.binding, self.name = binding, name

    def __call__(self, action, conversation=None):
        b = self.binding
        arguments = action.model_dump(mode="json")
        try:
            b.budget.tool()
        except BudgetExceeded:
            b.recorder.emit(
                "tool_not_executed", tool_name=self.name, arguments=arguments,
                reason=b.budget.reason, **b.recorder.pop_tool_link(self.name),
            )
            raise
        timeout = min(b.tool_timeout, b.budget.remaining())
        link = b.recorder.start_tool(
            self.name, arguments, effective_timeout_s=timeout,
            budget_at_execution=b.budget.snapshot(),
        )
        start = time.monotonic_ns()
        deadline = time.monotonic() + timeout
        connection = b.environment.connection
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        executor_start = time.monotonic_ns()
        error = None
        try:
            if self.name == "search":
                data = b.environment.search(action.query, action.top_k)
            elif self.name == "read_document":
                data = b.environment.read(
                    action.doc_id, start_sentence=action.start_sentence,
                    max_sentences=action.max_sentences,
                )
            else:
                data = b.environment.read(action.docid, offset=action.offset)
        except Exception as exc:
            error = exc
            data = {"error": str(exc), "error_type": type(exc).__name__}
        finally:
            executor_duration_ms = (time.monotonic_ns() - executor_start) / 1e6
            connection.set_progress_handler(None, 0)
        if error is None:
            if self.name == "search":
                b.recorder.retrieved_docids.update(r["docid"] for r in data)
            else:
                b.recorder.retrieved_docids.add(data["docid"])
        text = json.dumps(data, ensure_ascii=False)
        timed_out = error is not None and time.monotonic() >= deadline
        b.recorder.emit(
            "tool_end", tool_name=self.name, **link,
            executor_duration_ms=executor_duration_ms,
            round_trip_ms=(time.monotonic_ns() - start) / 1e6,
            outcome={
                "exit_code": 1 if error else 0, "timed_out": timed_out,
                "truncated": bool(data.get("truncated", False)) if isinstance(data, dict) else False,
                "clock_domain": "host-retrieval-monotonic",
            },
            error_type=type(error).__name__ if error else None,
            model_observation=text,
        )
        return RetrievalObservation.from_text(text, is_error=error is not None)


class SearchTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        return [
            cls(
                action_type=SearchAction,
                observation_type=RetrievalObservation,
                description=f"Search the fixed corpus. Return docid, title, URL and snippet. top_k must be <= {b.environment.config.retrieval.top_k}.",
                executor=RetrievalExecutor(b, cls.name),
            )
        ]


class ReadDocumentTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        return [
            cls(
                action_type=HotpotReadAction,
                observation_type=RetrievalObservation,
                description="Read original Wikipedia sentences by doc_id. Returns official title and original zero-based sentence IDs. Use these IDs in supporting_facts.",
                executor=RetrievalExecutor(b, cls.name),
            )
        ]


class GetDocumentTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        return [
            cls(
                action_type=BrowseReadAction,
                observation_type=RetrievalObservation,
                description="Read a fixed-corpus document page by docid. offset is a character offset. Follow next_offset when truncated is true.",
                executor=RetrievalExecutor(b, cls.name),
            )
        ]


for tool in [SearchTool, ReadDocumentTool, GetDocumentTool]:
    register_tool(tool.name, tool)


def make_tools(binding_key):
    b = get_binding(binding_key)
    read = ReadDocumentTool if b.environment.config.dataset.kind == "hotpot" else GetDocumentTool
    return [
        Tool(name=tool.name, params={"binding_key": binding_key}) for tool in [SearchTool, read]
    ]
