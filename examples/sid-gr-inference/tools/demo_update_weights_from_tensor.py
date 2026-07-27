#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo client: hot-swap model weights via colocate CUDA IPC (SGLang wire format).

Same orchestration as tools/demo_update_weights_from_disk.py, but the weight
transport is slime's chunked tensor push instead of a server-side disk read:

  - THIS client reads the HF checkpoint (safetensors), moves each chunk to the
    GPU, flattens it into a bucket and serializes it with ForkingPickler --
    CUDA tensors reduce to IPC *handles*, so the HTTP body stays tiny and the
    real weight memory is reconstructed by the server zero-copy via CUDA IPC.
  - Chunks are POSTed one at a time to /update_weights_from_tensor with
    load_format="flattened_bucket" and flush_cache=False, followed by one
    empty alignment bucket (mirrors slime's _empty_flattened_tensor_data).

REQUIREMENT: this client must run on the SAME GPU as the server (CUDA IPC is
intra-machine, per-device). The checkpoint path is read by the CLIENT (unlike
the disk demo, where the server reads it).

Flow (workload-verified like the disk demo):

   1. GET  /get_weight_version + /get_weights_by_name   (record "before" state)
   2. POST /generate                                    (output A: original weights)
   3. POST /pause_generation                            (mode=abort)
   4. POST /generate (short client timeout)             (probe: NO service while
      paused -- the call hangs and times out)
   5. GET  /flush_cache                                (retry until 200, slime-style)
   6. POST /update_weights_from_tensor  x N chunks      (+ one empty bucket)
   7. POST /continue_generation
   8. GET  /get_weight_version + /get_weights_by_name   (state changed)
   9. POST /generate                                    (output B: must DIFFER from A)
  10. pause -> flush -> chunked tensor update -> continue
                                                       (restore the ORIGINAL weights)
  11. GET  /get_weights_by_name                         (sample must match the
      original -- restore verified at the weight level)
  12. POST /generate                                    (output C: must EQUAL A)

Usage:

  python tools/perturb_checkpoint.py \
      --src /warehouse/Qwen3-1.7B --dst /models/Qwen3-1.7B-plus1 --delta 1.0

  # server started separately with GR_ALLOW_WEIGHT_UPDATE=1, then:
  python tools/demo_update_weights_from_tensor.py \
      --base-url http://127.0.0.1:8001 \
      --model-path /models/Qwen3-1.7B-plus1 --weight-version plus1 \
      --restore-model-path /warehouse/Qwen3-1.7B --restore-weight-version base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

# Make the repo's src importable when run as a plain script (we reuse the
# SGLang-compatible serializer the server also uses, so client and server are
# guaranteed to speak the same wire format).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from gr_inference.gr_serving.weight_ipc import (  # noqa: E402
    FlattenedTensorBucket,
    MultiprocessingSerializer,
)


_DETERMINISTIC_SAMPLING = {"temperature": 0.0, "ignore_eos": True}

# Reference workload (a real GR request shape), used verbatim for all probes:
#   input_ids: [9707, 887, 525, 263, 590, 34561, 13, 198, 9707, 887, 525, 263]
#   sampling_params: {temperature: 0, max_new_tokens: 3, ignore_eos: true, n: 256}
_PROBE_INPUT_IDS = (9707, 887, 525, 263, 590, 34561, 13, 198, 9707, 887, 525, 263)
_PROBE_MAX_NEW_TOKENS = 3
_PROBE_BEAM_WIDTH = 256


@dataclass(frozen=True)
class HTTPResult:
    status: int
    body: dict[str, Any]


class WeightSyncClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        query: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> HTTPResult:
        url = urljoin(self.base_url, path.lstrip("/"))
        if query:
            url = f"{url}?{urlencode(query)}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-GR-API-Key"] = self.api_key
        request = Request(url, data=body, method=method, headers=headers)
        try:
            with urlopen(  # noqa: S310
                request, timeout=timeout_s or self.timeout_s
            ) as response:
                return HTTPResult(
                    status=int(response.status),
                    body=_decode_body(response.read()),
                )
        except HTTPError as exc:
            return HTTPResult(status=int(exc.code), body=_decode_body(exc.read()))
        except URLError as exc:
            return HTTPResult(
                status=0,
                body={"error": {"code": "connection_error", "message": str(exc)}},
            )


def _decode_body(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw": raw[:256].decode("utf-8", errors="replace")}
    return decoded if isinstance(decoded, dict) else {"value": decoded}


def _require_ok(result: HTTPResult, step: str) -> dict[str, Any]:
    if result.status // 100 != 2:
        raise SystemExit(
            f"[{step}] failed: HTTP {result.status} "
            f"{json.dumps(result.body, ensure_ascii=False)[:400]}"
        )
    return result.body


def _weight_sample(
    client: WeightSyncClient, param_name: str, truncate_size: int
) -> list[float]:
    body = _require_ok(
        client.request(
            "GET",
            "/get_weights_by_name",
            query={"name": param_name, "truncate_size": truncate_size},
        ),
        "get_weights_by_name",
    )
    rows = body.get("parameter") or []
    if not rows:
        raise SystemExit(f"[get_weights_by_name] empty parameter for {param_name!r}")
    first_row = rows[0]
    flat = first_row if isinstance(first_row, list) else [first_row]
    return [float(v) for v in flat[:8]]


def _generate(
    client: WeightSyncClient,
    *,
    input_ids: Sequence[int],
    max_new_tokens: int,
    beam_width: int,
) -> list[int]:
    """Synchronous workload call; returns the top-beam output token ids.

    Deterministic (temperature=0, ignore_eos), so identical weights + inputs
    must yield identical output_ids across update rounds.
    """

    body = _require_ok(
        client.request(
            "POST",
            "/generate",
            {
                "request_id": f"demo-generate-{uuid.uuid4().hex[:8]}",
                "input_ids": list(input_ids),
                "sampling_params": {
                    **_DETERMINISTIC_SAMPLING,
                    "max_new_tokens": max_new_tokens,
                    "n": beam_width,
                },
            },
        ),
        "generate",
    )
    output_ids = body.get("output_ids")
    if not isinstance(output_ids, list) or not output_ids:
        raise SystemExit(
            f"[generate] unexpected response shape: "
            f"{json.dumps(body, ensure_ascii=False)[:400]}"
        )
    return [int(t) for t in output_ids]


def _probe_no_service_while_paused(
    client: WeightSyncClient,
    *,
    input_ids: Sequence[int],
    timeout_s: float = 3.0,
) -> None:
    """A synchronous /generate while paused must NOT complete.

    /generate submits the request and then waits server-side for the result;
    while paused the worker never ticks, so the call can only hang. A short
    client-side timeout turns that hang into the "no service" proof. The
    probe request is tiny (n=1) and runs harmlessly with the new weights once
    continue_generation resumes the worker.
    """

    try:
        result = client.request(
            "POST",
            "/generate",
            {
                "request_id": f"demo-paused-probe-{uuid.uuid4().hex[:8]}",
                "input_ids": list(input_ids),
                "sampling_params": {
                    **_DETERMINISTIC_SAMPLING,
                    "max_new_tokens": 1,
                    "n": 1,
                },
            },
            timeout_s=timeout_s,
        )
    except TimeoutError:
        result = HTTPResult(status=0, body={})
    if result.status == 0:
        print(
            f"   paused probe: /generate did not respond within {timeout_s}s "
            "-> no service while paused"
        )
        return
    if result.status // 100 == 2:
        raise SystemExit(
            "[paused probe] /generate SUCCEEDED while paused -- "
            "the engine kept serving after pause_generation"
        )
    raise SystemExit(
        f"[paused probe] unexpected HTTP {result.status}: "
        f"{json.dumps(result.body, ensure_ascii=False)[:400]}"
    )


def _flush_cache_with_retry(client: WeightSyncClient, retries: int) -> None:
    for attempt in range(1, retries + 1):
        flushed = client.request("GET", "/flush_cache")
        if flushed.status == 200:
            print(f"   flush_cache: {flushed.body} (attempt {attempt})")
            return
        time.sleep(1.0)
    raise SystemExit(
        f"[flush_cache] still busy after {retries} attempts: "
        f"{json.dumps(flushed.body, ensure_ascii=False)[:400]}"
    )


# --------------------------------------------------------------------------- #
# Chunked CUDA-IPC weight push (slime colocate transport)
# --------------------------------------------------------------------------- #


def _checkpoint_tensor_names(model_path: str) -> list[tuple[str, str]]:
    """Return (tensor_name, shard_file) pairs for a HF safetensors checkpoint."""

    root = Path(model_path)
    shards = sorted(p for p in root.glob("*.safetensors") if p.is_file())
    if not shards:
        raise SystemExit(f"[tensor push] no *.safetensors under {model_path}")
    pairs: list[tuple[str, str]] = []
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                pairs.append((name, str(shard)))
    return pairs


def _serialize_bucket_payload(named_tensors: list[tuple[str, Any]]) -> str:
    """Build a flattened bucket and serialize it (CUDA tensors -> IPC handles)."""

    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    return MultiprocessingSerializer.serialize(
        {"flattened_tensor": bucket.flattened_tensor, "metadata": bucket.metadata},
        output_str=True,
    )


def _post_tensor_bucket(
    client: WeightSyncClient, payload_b64: str, *, weight_version: str, label: str
) -> None:
    result = _require_ok(
        client.request(
            "POST",
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": [payload_b64],
                "load_format": "flattened_bucket",
                "flush_cache": False,
                "weight_version": weight_version,
            },
        ),
        f"update_weights_from_tensor({label})",
    )
    print(f"   update_weights_from_tensor[{label}]: {result}")


def _push_weights_via_ipc(
    client: WeightSyncClient,
    *,
    model_path: str,
    weight_version: str,
    chunk_size: int,
    device: str,
) -> None:
    """Push a checkpoint to the server in chunks over CUDA IPC (slime-style)."""

    pairs = _checkpoint_tensor_names(model_path)
    chunks = [
        pairs[index : index + chunk_size]
        for index in range(0, len(pairs), chunk_size)
    ]
    print(f"   pushing {len(pairs)} tensors in {len(chunks)} chunk(s) via CUDA IPC")
    for chunk_index, chunk in enumerate(chunks):
        named_tensors: list[tuple[str, Any]] = []
        by_shard: dict[str, list[str]] = {}
        for name, shard in chunk:
            by_shard.setdefault(shard, []).append(name)
        for shard, names in by_shard.items():
            with safe_open(shard, framework="pt") as handle:
                for name in names:
                    named_tensors.append(
                        (name, handle.get_tensor(name).to(device))
                    )
        # Keep the CUDA tensors alive across the synchronous POST: the payload
        # carries IPC *handles*, so the server reads our GPU memory during the
        # call. Freeing only after the response mirrors slime's flow.
        payload = _serialize_bucket_payload(named_tensors)
        _post_tensor_bucket(
            client,
            payload,
            weight_version=weight_version,
            label=f"chunk {chunk_index + 1}/{len(chunks)}",
        )
        del named_tensors, payload
        torch.cuda.empty_cache()

    # slime also sends empty alignment buckets; the server treats them as no-ops.
    empty_payload = MultiprocessingSerializer.serialize(
        {
            "flattened_tensor": torch.empty(0, dtype=torch.uint8, device=device),
            "metadata": [],
        },
        output_str=True,
    )
    _post_tensor_bucket(
        client, empty_payload, weight_version=weight_version, label="empty bucket"
    )


def _update_round(
    client: WeightSyncClient,
    *,
    model_path: str,
    weight_version: str,
    flush_retries: int,
    chunk_size: int,
    device: str,
    label: str,
) -> None:
    """pause_generation -> flush_cache -> chunked tensor push -> continue."""

    paused = _require_ok(
        client.request("POST", "/pause_generation", {}), f"pause_generation({label})"
    )
    print(f"   pause_generation: {paused}")
    _flush_cache_with_retry(client, flush_retries)
    _push_weights_via_ipc(
        client,
        model_path=model_path,
        weight_version=weight_version,
        chunk_size=chunk_size,
        device=device,
    )
    resumed = _require_ok(
        client.request("POST", "/continue_generation", {}),
        f"continue_generation({label})",
    )
    print(f"   continue_generation: {resumed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        required=True,
        help="NEW checkpoint dir, read by THIS client (same GPU host as server)",
    )
    parser.add_argument(
        "--weight-version",
        default=None,
        help="version label for the new weights (default: checkpoint dir name)",
    )
    parser.add_argument(
        "--restore-model-path",
        required=True,
        help="ORIGINAL checkpoint dir (restored in the second update round)",
    )
    parser.add_argument(
        "--restore-weight-version",
        default=None,
        help="version label for the restore round (default: version recorded "
        "at startup, else the restore dir name)",
    )
    parser.add_argument(
        "--param-name",
        default="embed_tokens.weight",
        help="module parameter name to sample before/after",
    )
    parser.add_argument("--truncate-size", type=int, default=2)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--flush-retries",
        type=int,
        default=60,
        help="flush_cache retries while requests are in flight (slime uses 60)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="tensors per IPC bucket POST (slime pushes per-bucket chunks)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="GPU this client stages tensors on; must be the server's GPU",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: the tensor path shares GPU memory via IPC")

    weight_version = args.weight_version or args.model_path.rstrip("/").split("/")[-1]
    client = WeightSyncClient(args.base_url, api_key=args.api_key)
    input_ids = list(_PROBE_INPUT_IDS)

    print(f"== target: {args.base_url} (staging device: {args.device})")

    # --- 1. record the "before" state -------------------------------------
    before_version = _require_ok(
        client.request("GET", "/get_weight_version"), "get_weight_version"
    )
    print(f"1. weight_version (before): {before_version}")
    before_sample = _weight_sample(client, args.param_name, args.truncate_size)
    print(f"   {args.param_name} (before): {before_sample}")

    # --- 2. workload output with the ORIGINAL weights ----------------------
    output_a = _generate(
        client,
        input_ids=input_ids,
        max_new_tokens=_PROBE_MAX_NEW_TOKENS,
        beam_width=_PROBE_BEAM_WIDTH,
    )
    print(f"2. generate (original weights): output_ids={output_a}")

    # --- 3-7. pause (prove no service) -> flush -> IPC push -> continue ----
    print("3. update round 1: push the new weights via CUDA IPC")
    paused = _require_ok(
        client.request("POST", "/pause_generation", {}), "pause_generation"
    )
    print(f"   pause_generation: {paused}")
    _probe_no_service_while_paused(client, input_ids=input_ids)
    _flush_cache_with_retry(client, args.flush_retries)
    _push_weights_via_ipc(
        client,
        model_path=args.model_path,
        weight_version=weight_version,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    resumed = _require_ok(
        client.request("POST", "/continue_generation", {}), "continue_generation"
    )
    print(f"   continue_generation: {resumed}")

    # --- 8. confirm state changed -------------------------------------------
    after_version = _require_ok(
        client.request("GET", "/get_weight_version"), "get_weight_version"
    )
    print(f"8. weight_version (after): {after_version}")
    if after_version.get("weight_version") != weight_version:
        raise SystemExit(
            f"weight_version mismatch: expected {weight_version!r}, "
            f"got {after_version.get('weight_version')!r}"
        )
    after_sample = _weight_sample(client, args.param_name, args.truncate_size)
    print(f"   {args.param_name} (after): {after_sample}")

    # --- 9. workload output with the NEW weights: must differ ---------------
    output_b = _generate(
        client,
        input_ids=input_ids,
        max_new_tokens=_PROBE_MAX_NEW_TOKENS,
        beam_width=_PROBE_BEAM_WIDTH,
    )
    print(f"9. generate (new weights): output_ids={output_b}")
    if output_b == output_a:
        raise SystemExit(
            "output UNCHANGED after the weight swap -- "
            "the new weights do not seem to be in effect"
        )
    print("   -> output changed after the weight swap (as expected)")

    # --- 10. restore the ORIGINAL weights ------------------------------------
    restore_version = (
        args.restore_weight_version
        or before_version.get("weight_version")
        or args.restore_model_path.rstrip("/").split("/")[-1]
    )
    print("10. update round 2: restore the original weights via CUDA IPC")
    _update_round(
        client,
        model_path=args.restore_model_path,
        weight_version=str(restore_version),
        flush_retries=args.flush_retries,
        chunk_size=args.chunk_size,
        device=args.device,
        label="restore",
    )

    # --- 11. weight sample after restore: must match the original sample ----
    restored_sample = _weight_sample(client, args.param_name, args.truncate_size)
    print(f"11. {args.param_name} (restored): {restored_sample}")
    if restored_sample != before_sample:
        raise SystemExit(
            f"weight sample DIFFERS from the original after restore -- "
            f"expected {before_sample}, got {restored_sample}"
        )
    print("   -> weight sample matches the original (restore verified)")

    # --- 12. workload output after restore: must match the original ---------
    output_c = _generate(
        client,
        input_ids=input_ids,
        max_new_tokens=_PROBE_MAX_NEW_TOKENS,
        beam_width=_PROBE_BEAM_WIDTH,
    )
    print(f"12. generate (restored weights): output_ids={output_c}")
    if output_c != output_a:
        raise SystemExit(
            "output DIFFERS from the original after restoring weights -- "
            f"expected {output_a}, got {output_c}"
        )
    print("   -> output matches the original weights exactly (restore verified)")

    print("OK: colocate (CUDA IPC) weight update + restore completed and verified.")


if __name__ == "__main__":
    main()
