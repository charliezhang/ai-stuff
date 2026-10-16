"""Verify data by hashing token *contents*, not the .pt file bytes.

slowrun's EXPECTED_HASHES are over the torch.save() file, which differs across
environments even with identical tokens (we confirmed text order matches FineWeb
sample-10BT row-for-row). Content hashes below are from our 2026-10-15 prep.
"""
import hashlib
import sys
import torch

CONTENT_HASHES = {
    "fineweb_val.pt": "6b2ce462cfd36984db3d569588051206a4543b7d637aea1a002cca0c80fd97b4",
    "fineweb_train.pt": "ced9b895ebc8fda269e4ae09af176fc7df7562b0356559f1202dd9c264a8faea",
}


def content_hash(path):
    d = torch.load(path)
    h = hashlib.sha256()
    h.update(d["tokens"].numpy().tobytes())
    h.update(d["doc_starts"].numpy().tobytes())
    h.update(f'{d["bos_id"]},{d["seq_shuffle_seed"]},{d["seq_size"]}'.encode())
    return h.hexdigest()


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    for name, expected in CONTENT_HASHES.items():
        actual = content_hash(f"{data_dir}/{name}")
        status = "unset" if expected is None else ("OK" if actual == expected else "MISMATCH")
        print(f"{name}: {actual} [{status}]")
        assert status != "MISMATCH"
