"""Replay recorded tool calls without an LLM; retain response hashes and timings."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx
from fastmcp import Client
from mcp.types import TextContent


def fingerprint(text):
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True)
    except ValueError:
        pass
    return hashlib.sha256(text.encode()).hexdigest()


def observation(result):
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    return '\n'.join(texts) if texts else json.dumps(result.data, ensure_ascii=False)


def row(call, repetition, elapsed, text, timing=None):
    digest = fingerprint(text)
    if digest != call['expected_sha256']:
        raise ValueError(f'Replay changed the response for operation {call["operation"]}')
    return {'operation': call['operation'], 'name': call['name'], 'repetition': repetition,
            'round_trip_ms': elapsed, 'response_sha256': digest,
            'model_observation_bytes': len(text.encode()), 'server_timing': timing}


def hotpot(args, calls):
    rows, ping = [], []
    with httpx.Client(base_url=args.url, trust_env=False, timeout=30) as client:
        for _ in range(20):
            started = time.perf_counter_ns()
            response = client.post('/info', json={'timeout_seconds': 30})
            response.raise_for_status()
            response.json()
            ping.append((time.perf_counter_ns() - started) / 1e6)
        for repetition in range(args.repeats + 1):
            for call in calls:
                arguments = dict(call['arguments'])
                # SDK action metadata is not a retrieval RPC argument.
                arguments.pop('kind', None)
                path = '/search'
                if call['name'] == 'read_document':
                    path = '/read'
                    arguments['docid'] = arguments.pop('doc_id')
                started = time.perf_counter_ns()
                response = client.post(path, json={**arguments, 'timeout_seconds': 30})
                response.raise_for_status()
                body = response.json()
                elapsed = (time.perf_counter_ns() - started) / 1e6
                text = json.dumps(body['result'], ensure_ascii=False)
                item = row(call, repetition, elapsed, text, body['timing'])
                if repetition:
                    rows.append(item)
    return rows, ping


async def browsecomp(args, calls):
    rows, ping = [], []
    async with Client(args.url, timeout=30) as client:
        for _ in range(20):
            started = time.perf_counter_ns()
            await client.list_tools()
            ping.append((time.perf_counter_ns() - started) / 1e6)
        for repetition in range(args.repeats + 1):
            for call in calls:
                started = time.perf_counter_ns()
                result = await client.call_tool(call['name'], call['arguments'])
                text = observation(result)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                item = row(call, repetition, elapsed, text)
                if repetition:
                    rows.append(item)
    return rows, ping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['hotpot', 'browsecomp'], required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    calls = json.loads(args.requests.read_text())
    rows, ping = hotpot(args, calls) if args.backend == 'hotpot' else asyncio.run(browsecomp(args, calls))
    report = {'backend': args.backend, 'url': args.url, 'repeats': args.repeats,
              'warmup_pass_excluded': True, 'ordered_response_hashes_verified': True,
              'control_call': 'Hotpot info' if args.backend == 'hotpot' else 'MCP list_tools',
              'control_call_ms': ping, 'measurements': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'backend': args.backend, 'measured_calls': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
