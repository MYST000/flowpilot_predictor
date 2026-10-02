"""Direct Brave search and uncached page reads for the paired live-web protocol."""

import hashlib
import ipaddress
import json
import os
import re
import socket
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.parse import urldefrag, urlsplit

import httpx
from bs4 import BeautifulSoup

SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def live_task(task, kind):
    question = task.instruction.split("\n\nQuestion: ", 1)[1]
    if kind == "hotpot":
        instruction = (
            "Answer using live Wikipedia search and page reads. Finish with JSON containing "
            '"answer" (string) and "supporting_facts" (list of [page title, sentence ID]). '
            "Sentence IDs refer to this live page extraction, not the historical dataset."
        )
    else:
        instruction = (
            "Answer using live Internet search and page reads. Finish with a concise answer "
            "and supporting page URLs. Do not consult benchmark question/answer mirrors."
        )
    return replace(
        task,
        instruction=instruction + "\n\nQuestion: " + question,
        public_metadata={**task.public_metadata, "retrieval_protocol": "live-web-brave-v1"},
    )


def public_url(url):
    url = urldefrag(url)[0]
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Expected a public HTTP(S) page URL")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Only standard web ports are allowed")
    addresses = socket.getaddrinfo(
        parsed.hostname,
        parsed.port or (443 if parsed.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise ValueError("Page URL resolves to a non-public address")
    return url


class LiveWebEnvironment:
    def __init__(self, config, *, transport=None):
        self.config = config
        self._transport = transport
        self.client = None
        self.deadline = None
        self.last_http_attempts = []
        self.known_urls = set()
        self.tool_definitions = {
            "search": SimpleNamespace(
                description="Search the live Internet. Return docid (page URL), title, URL and snippet. Use returned docid in get_document."
            ),
            "get_document": SimpleNamespace(
                description="Fetch a live page using a docid URL returned by search. This performs a fresh HTTP request; returns extracted text and content identity."
            ),
        }

    def prepare(self):
        if not os.environ.get(self.config.retrieval.web_api_key_env):
            raise ValueError(
                "Missing Brave search credential: " + self.config.retrieval.web_api_key_env
            )
        if self.client is None:
            self.client = httpx.Client(
                trust_env=False,
                follow_redirects=False,
                transport=self._transport,
                headers={
                    "User-Agent": "FlowPilotResearch/1.0 (paired live retrieval experiment)",
                    "Accept-Encoding": "identity",
                },
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            )
        return {
            "backend": "live_web",
            "provider": "brave",
            "search_endpoint": SEARCH_URL,
            "corpus_revision": self.config.retrieval.corpus_revision,
            "cache_namespace": self.config.retrieval.corpus_revision,
            "network_route": "server-direct",
            "trust_env": False,
            "local_response_cache": False,
            "automatic_retries": 0,
            "validation_scope": "credential_presence_and_client_only; use network preflight separately",
            "country": self.config.retrieval.web_country,
            "search_lang": self.config.retrieval.web_search_lang,
        }

    @contextmanager
    def operation_timeout(self, seconds):
        self.last_http_attempts = []
        self.deadline = time.monotonic() + seconds
        try:
            yield
            if time.monotonic() > self.deadline:
                raise TimeoutError("Live web tool deadline exceeded")
        finally:
            self.deadline = None

    def _remaining(self):
        remaining = (
            (self.deadline - time.monotonic())
            if self.deadline is not None
            else self.config.runtime.tool_timeout
        )
        if remaining <= 0:
            raise TimeoutError("Live web tool deadline exceeded")
        return remaining

    def _get(self, url, *, params=None, api=False):
        assert self.client is not None
        headers = (
            {
                "X-Subscription-Token": os.environ[self.config.retrieval.web_api_key_env],
                "Accept": "application/json",
            }
            if api
            else {}
        )
        for redirect in range(6):
            start = time.monotonic_ns()
            record = {
                "attempt_index": len(self.last_http_attempts),
                "url": url,
                "kind": "search" if api else "page",
                "redirect_index": redirect,
                "started_at": datetime.now(UTC).isoformat(),
                "status_code": None,
                "cache_namespace": self.config.retrieval.corpus_revision,
                "proxy_used": False,
                "dns_ms": None,
                "tcp_ms": None,
                "tls_ms": None,
                "ttfb_ms": None,
                "connection_reused": None,
                "response_bytes": 0,
            }
            try:
                if not api:
                    url = public_url(url)
                    if (
                        self.config.dataset.kind == "hotpot"
                        and urlsplit(url).hostname != "en.wikipedia.org"
                    ):
                        raise ValueError("Hotpot live protocol only reads en.wikipedia.org")
                with self.client.stream(
                    "GET", url, params=params, headers=headers, timeout=self._remaining()
                ) as response:
                    record.update(
                        status_code=response.status_code,
                        url=str(response.url),
                        content_type=response.headers.get("content-type"),
                        etag=response.headers.get("etag"),
                        last_modified=response.headers.get("last-modified"),
                        cache_control=response.headers.get("cache-control"),
                        retry_after=response.headers.get("retry-after"),
                    )
                    if response.is_redirect:
                        if api:
                            raise ValueError(
                                "Search API redirects are disabled to protect credentials"
                            )
                        url = str(response.url.join(response.headers["location"]))
                        params = None
                        continue
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        self._remaining()
                        size += len(chunk)
                        record["response_bytes"] = size
                        if size > self.config.retrieval.web_max_response_bytes:
                            raise ValueError("HTTP response exceeds live-web byte budget")
                        chunks.append(chunk)
                    body = b"".join(chunks)
                    record["content_sha256"] = hashlib.sha256(body).hexdigest()
                    return (
                        body,
                        response.encoding or "utf-8",
                        str(response.url),
                        record["content_type"] or "",
                    )
            except httpx.TimeoutException as exc:
                record["error_type"] = type(exc).__name__
                raise TimeoutError("Direct HTTP request timed out") from exc
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                raise
            finally:
                record["http_attempt_ms"] = (time.monotonic_ns() - start) / 1e6
                self.last_http_attempts.append(record)
        raise ValueError("Too many page redirects")

    def search(self, query, top_k):
        cfg = self.config.retrieval
        if not 1 <= top_k <= cfg.top_k:
            raise ValueError("top_k exceeds live-web profile limit")
        query = query.strip()
        if not query:
            raise ValueError("Empty search query")
        if self.config.dataset.kind == "hotpot":
            query = f"({query}) site:en.wikipedia.org"
        body, _, _, _ = self._get(
            SEARCH_URL,
            api=True,
            params={
                "q": query,
                "count": top_k,
                "country": cfg.web_country,
                "search_lang": cfg.web_search_lang,
                "safesearch": "moderate",
                "text_decorations": "false",
            },
        )
        rows = []
        for item in json.loads(body).get("web", {}).get("results", []):
            url = urldefrag(item["url"])[0]
            if urlsplit(url).scheme not in {"http", "https"}:
                continue
            if (
                self.config.dataset.kind == "hotpot"
                and urlsplit(url).hostname != "en.wikipedia.org"
            ):
                continue
            self.known_urls.add(url)
            rows.append(
                {
                    "docid": url,
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": BeautifulSoup(item.get("description", ""), "html.parser").get_text(
                        " ", strip=True
                    )[: cfg.snippet_chars],
                }
            )
        return rows[:top_k]

    def read(self, docid, *, offset=0, start_sentence=0, max_sentences=20):
        docid = urldefrag(docid)[0]
        if docid not in self.known_urls:
            raise ValueError("Use a docid URL returned by search in this task")
        body, encoding, url, content_type = self._get(docid)
        if not (
            "text/html" in content_type
            or "text/plain" in content_type
            or "application/xhtml+xml" in content_type
        ):
            raise ValueError("Live-web v1 supports HTML and plain text pages only")
        soup = BeautifulSoup(body.decode(encoding, errors="replace"), "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text(" ", strip=True)
        common = {
            "docid": docid,
            "url": url,
            "title": title,
            "content_sha256": hashlib.sha256(body).hexdigest(),
            "fetched_at": datetime.now(UTC).isoformat(),
            "text_extractor": "bs4-html-v1",
        }
        if self.config.dataset.kind == "hotpot":
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if start_sentence < 0 or not 1 <= max_sentences <= 100:
                raise ValueError("Invalid sentence range")
            selected, count = [], 0
            for i in range(start_sentence, min(len(sentences), start_sentence + max_sentences)):
                sentence = sentences[i]
                if count + len(sentence) > self.config.retrieval.read_chars and selected:
                    break
                selected.append([i, sentence[: self.config.retrieval.read_chars]])
                count += len(selected[-1][1])
            end = start_sentence + len(selected)
            return {
                **common,
                "sentences": selected,
                "next_sentence": end if end < len(sentences) else None,
                "sentence_identity": "live-extraction-not-official-hotpot",
            }
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        end = min(len(text), offset + self.config.retrieval.read_chars)
        return {
            **common,
            "text": text[offset:end],
            "offset": offset,
            "truncated": end < len(text),
            "next_offset": end if end < len(text) else None,
        }

    def call_tool(self, name, arguments):
        data = (
            self.search(arguments["query"], self.config.retrieval.top_k)
            if name == "search"
            else self.read(arguments["docid"])
        )
        return SimpleNamespace(data=data, text=json.dumps(data, ensure_ascii=False))

    def quiesce(self):
        pass

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
