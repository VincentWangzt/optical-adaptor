"""A local OpenAI-style chat endpoint for the same standalone inference backends."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from optical_adaptor.benchmark.config import load_benchmark
from optical_adaptor.inference.backend import (
    BACKENDS,
    GREEDY_DECODING,
    ChatRequest,
    build_backend,
)
from optical_adaptor.training.config import load_credentials


def completion(backend, payload, default_seed):
    allowed = {"model", "messages", "max_tokens", "temperature", "seed", "stream", "n"}
    if set(payload) - allowed:
        raise ValueError(f"unsupported request fields: {sorted(set(payload) - allowed)}")
    if (
        payload.get("temperature", 0) != 0
        or payload.get("stream", False)
        or payload.get("n", 1) != 1
    ):
        raise ValueError("this endpoint supports greedy, non-streaming, single-choice requests")
    response = backend.generate(
        [
            ChatRequest(
                messages=payload["messages"],
                max_tokens=payload["max_tokens"],
                seed=payload.get("seed", default_seed),
                decoding=GREEDY_DECODING,
            )
        ]
    )[0]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", backend.identity["backend"]),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.text},
                "finish_reason": response.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.prompt_tokens + response.completion_tokens,
        },
        "optical": response.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    config, pipeline, _ = load_benchmark(args.config)
    load_credentials(pipeline, wandb=False)
    backend = build_backend(
        args.backend,
        pipeline,
        config.backend,
        checkpoint=pipeline.repo / config.checkpoint,
        seed=config.seed,
    )

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ok", "backend": backend.identity})
            elif self.path == "/v1/models":
                self.respond(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": args.backend, "object": "model", "owned_by": "local"}],
                    },
                )
            else:
                self.respond(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 128 * 1024 * 1024:
                    raise ValueError("request must be nonempty and at most 128 MiB")
                payload = json.loads(self.rfile.read(length))
                result = completion(backend, payload, config.seed)
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, {"error": str(error)})
                return
            self.respond(200, result)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving {args.backend} at http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        backend.close()


if __name__ == "__main__":
    main()
