"""Run a JSONL task dataset against an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.tasks.validators import load_task, validate_answer, validate_dataset


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def complete(self, payload: dict[str, Any]) -> tuple[dict, float]:
        request_body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            request = urllib.request.Request(
                self.endpoint, data=request_body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data, time.perf_counter() - started
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {detail[:1000]}")
                if 400 <= exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for record in _read_jsonl(path):
        if record.get("status") == "ok":
            completed.add(record["task_id"])
    return completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_signature(config: dict, dataset_path: Path) -> dict:
    return {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "model": config["model"],
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get("max_tokens", 16),
        "seed": config.get("seed"),
        "extra_body": config.get("extra_body", {}),
    }


def _ensure_result_manifest(output_path: Path, signature: dict) -> None:
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise ValueError(
                f"experiment settings differ from existing {manifest_path}; "
                "use a new output path"
            )
        return
    if output_path.exists() and output_path.stat().st_size:
        raise ValueError(
            f"existing result file {output_path} has no manifest; use a new output path"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(signature, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
    os.replace(temporary_path, manifest_path)


def _response_text(response: dict) -> tuple[str, str | None, dict]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"response has no choices: {response}")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise TypeError("response message content is not a string")
    return content, choice.get("finish_reason"), response.get("usage") or {}


def _run_one(record: dict, config: dict, client: ChatClient) -> dict:
    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": [{"role": "user", "content": record["prompt"]}],
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get("max_tokens", 16),
    }
    if config.get("seed") is not None:
        payload["seed"] = config["seed"]
    payload.update(config.get("extra_body", {}))
    response, latency = client.complete(payload)
    output, finish_reason, usage = _response_text(response)
    task = load_task(record)
    score = validate_answer(task, output)
    return {
        "status": "ok",
        "task_id": record["task_id"],
        "task_family": record["task_family"],
        "work": record.get("work"),
        "depth": record["depth"],
        "model": config["model"],
        "output": output,
        **score,
        "finish_reason": finish_reason,
        "usage": usage,
        "latency_seconds": round(latency, 6),
    }


def run(config: dict, limit: int | None = None, dry_run: bool = False) -> int:
    dataset_path = Path(config["dataset"])
    output_path = Path(config["output"])
    validation = validate_dataset(str(dataset_path))
    if validation["invalid"]:
        raise ValueError(
            f"dataset validation failed for {len(validation['invalid'])} records"
        )

    records = _read_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]
    if not dry_run:
        _ensure_result_manifest(
            output_path, _experiment_signature(config, dataset_path)
        )
    completed = _completed_ids(output_path)
    pending = [record for record in records if record["task_id"] not in completed]
    print(
        f"dataset={dataset_path} valid={validation['valid']} "
        f"selected={len(records)} completed={len(records) - len(pending)} "
        f"pending={len(pending)}"
    )
    if dry_run or not pending:
        return 0

    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    client = ChatClient(
        base_url=config["base_url"],
        api_key=os.environ.get(api_key_env),
        timeout_seconds=float(config.get("timeout_seconds", 180)),
        retries=int(config.get("retries", 3)),
    )
    concurrency = int(config.get("concurrency", 8))
    failures: list[tuple[str, str]] = []
    write_lock = threading.Lock()
    finished = 0

    with (
        output_path.open("a", encoding="utf-8") as output_file,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        futures = {
            executor.submit(_run_one, record, config, client): record["task_id"]
            for record in pending
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate individual API failures
                failures.append((task_id, str(exc)))
            else:
                with write_lock:
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
            finished += 1
            if finished % 10 == 0 or finished == len(pending):
                print(
                    f"progress={finished}/{len(pending)} failures={len(failures)}",
                    flush=True,
                )

    if failures:
        print("failed task requests:", file=sys.stderr)
        for task_id, error in failures[:20]:
            print(f"  {task_id}: {error}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
        print("rerun the same command to retry failed tasks", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.json")
    parser.add_argument("--model", help="override the served model name")
    parser.add_argument("--base-url", help="override the API base URL")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--limit", type=int, help="run only the first N records")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)
    for key in ("model", "base_url", "concurrency"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return run(config, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
