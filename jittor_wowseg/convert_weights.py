#!/usr/bin/env python
"""Convert the released WOW-Seg safetensors checkpoint to a Jittor pickle.

The conversion is intentionally lossless apart from BF16 -> FP16, which is
needed by Jittor's CUDA kernels.  It does not rename parameters, so the
runtime can validate every expected tensor before inference starts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


SIGNATURE = b"HCAJSLHD"
FORMAT = "wowseg-jittor-v1"


def save_jittor_pickle(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        # Do not materialize a second ~2 GB serialized copy in RAM. Jittor's
        # pickle format is a normal pickle followed by SHA1 and this signature.
        class HashingWriter:
            def __init__(self, stream):
                self.stream = stream
                self.digest = hashlib.sha1()

            def write(self, data):
                self.digest.update(data)
                return self.stream.write(data)

            def flush(self):
                return self.stream.flush()

        writer = HashingWriter(handle)
        pickle.dump(payload, writer, protocol=4)
        handle.write(writer.digest.digest())
        handle.write(SIGNATURE)


def sha256_file(path: Path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def convert_checkpoint(input_path: Path, output_path: Path):
    tensors = {}
    with safe_open(str(input_path), framework="pt", device="cpu") as archive:
        keys = list(archive.keys())
        for index, key in enumerate(keys, 1):
            value = archive.get_tensor(key)
            if value.dtype == torch.bfloat16:
                value = value.to(torch.float16)
            tensors[key] = np.ascontiguousarray(value.numpy())
            if index % 100 == 0 or index == len(keys):
                print(f"converted {index}/{len(keys)} tensors", flush=True)
    payload = {
        "format": FORMAT,
        "source": input_path.name,
        "source_sha256": sha256_file(input_path),
        "state_dict": tensors,
    }
    save_jittor_pickle(payload, output_path)
    manifest = {
        "format": FORMAT,
        "source": str(input_path),
        "output": str(output_path),
        "source_sha256": payload["source_sha256"],
        "tensor_count": len(tensors),
        "parameter_count": int(sum(value.size for value in tensors.values())),
    }
    del payload, tensors
    gc.collect()
    return manifest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    model_dir = root / "jittor_wowseg" / "weights" / "wow_seg"
    parser.add_argument("--input", type=Path, default=model_dir / "model.safetensors")
    parser.add_argument("--output", type=Path, default=model_dir / "wow_seg.pkl")
    args = parser.parse_args()
    manifest = convert_checkpoint(args.input.resolve(), args.output.resolve())
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
