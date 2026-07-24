#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Perturb a HuggingFace checkpoint: add a constant to every weight tensor,
re-saving a complete, structurally-identical checkpoint to a new directory.

Used to generate a visibly-different checkpoint for weight hot-update testing
(update_weights_from_disk): a +1.0 shift is exactly representable in bf16/fp16/fp32
and trivially detectable via /get_weights_by_name (every value differs by ~1.0),
so you can confirm an update actually took effect. Architecture/config is
unchanged, so the engine's structural-compat check still passes.

Usage:
  python tools/perturb_checkpoint.py \
      --src /warehouse/Qwen3-1.7B --dst /warehouse/Qwen3-1.7B-plus1 --delta 1.0

Then drive the engine:
  curl -X POST http://127.0.0.1:8000/update_weights_from_disk \
       -d '{"model_path":"/warehouse/Qwen3-1.7B-plus1","weight_version":"plus1"}'
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def perturb(src: str | Path, dst: str | Path, delta: float) -> int:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(p.name for p in src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors weight files under {src}")

    # 1) Copy everything except the weight shards (config.json, the safetensors
    #    index, tokenizer files, generation_config, ...) so the output is a
    #    complete HF checkpoint. The index's weight_map still matches because we
    #    keep the same shard filenames and tensor names.
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, dst / f.name)

    # 2) Re-emit each shard with `delta` added to every tensor (dtype preserved;
    #    1.0 is exact in bf16/fp16/fp32, so the shift is lossless and detectable).
    total = 0
    for shard in shards:
        tensors = load_file(str(src / shard))
        for tensor in tensors.values():
            tensor.add_(delta)
        save_file(tensors, str(dst / shard))
        total += len(tensors)
        print(f"  {shard}: {len(tensors)} tensors +{delta}")

    print(f"wrote {total} tensors (+{delta}) -> {dst}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/warehouse/Qwen3-1.7B")
    ap.add_argument("--dst", required=True, help="output checkpoint directory")
    ap.add_argument("--delta", type=float, default=1.0)
    args = ap.parse_args()
    perturb(args.src, args.dst, args.delta)


if __name__ == "__main__":
    main()
