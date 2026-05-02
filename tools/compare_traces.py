"""Compare two MacFleet training traces for forensic verification.

Run after both Macs finish two_mac_traced_train.py. Reads both JSONL
trace files and verifies row-by-row that distributed training did
exactly what was intended:

  1. Each rank processed DIFFERENT sample indices (sampler split worked)
  2. Each rank computed DIFFERENT local gradients (different inputs)
  3. After allreduce both ranks held the SAME averaged gradient
  4. After optimizer.step(), parameters MATCH on both ranks
  5. Bytes sent on rank N ≈ bytes received on rank 1-N (no traffic was
     local — every gradient byte was actually exchanged over the wire)

Usage:
    python3 tools/compare_traces.py <trace_rank0.jsonl> <trace_rank1.jsonl>

Exit code 0 = all checks passed, 1 = any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_trace(path: str) -> tuple[dict, list[dict], dict]:
    """Returns (header, [step_records], footer). Raises on missing/malformed."""
    header = None
    footer = None
    steps: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("type")
            if t == "header":
                header = rec
            elif t == "footer":
                footer = rec
            elif t == "step":
                steps.append(rec)
            else:
                raise ValueError(f"unknown record type {t!r} in {path}")
    if header is None or footer is None:
        raise ValueError(f"{path} missing header or footer (run did not finish?)")
    return header, steps, footer


def color(text: str, ok: bool) -> str:
    return f"\033[32m{text}\033[0m" if ok else f"\033[31m{text}\033[0m"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    path0, path1 = sys.argv[1], sys.argv[2]
    h0, steps0, f0 = load_trace(path0)
    h1, steps1, f1 = load_trace(path1)

    # Sanity: ranks must be 0 and 1 respectively
    if h0["rank"] != 0:
        print(f"ERROR: {path0} is rank {h0['rank']}, expected 0")
        return 1
    if h1["rank"] != 1:
        print(f"ERROR: {path1} is rank {h1['rank']}, expected 1")
        return 1

    print(f"=== TRACE COMPARISON ===")
    print(f"  rank 0 file: {path0}  ({len(steps0)} step records)")
    print(f"  rank 1 file: {path1}  ({len(steps1)} step records)")
    print()
    print(f"  config: world_size={h0['world_size']}, epochs={h0['epochs']}, "
          f"batch={h0['batch_size']}, params={h0['n_params']:,}")
    print()

    # Step counts must match
    if len(steps0) != len(steps1):
        print(color(
            f"FAIL: step counts differ ({len(steps0)} vs {len(steps1)})", False,
        ))
        return 1

    # Per-step checks
    local_grad_diff_count = 0
    synced_grad_match_count = 0
    params_match_count = 0
    batch_diff_count = 0
    issues: list[str] = []

    for s, (a, b) in enumerate(zip(steps0, steps1)):
        # Check 1: batch indices differ (sampler gave each rank distinct samples)
        if set(a["batch_indices"]) != set(b["batch_indices"]):
            batch_diff_count += 1
        else:
            issues.append(
                f"step {s}: batch_indices IDENTICAL — both ranks "
                f"processed same samples (sampler bug?)"
            )

        # Check 2: local gradients differ (different samples → different grads)
        if a["local_grad"]["sha1"] != b["local_grad"]["sha1"]:
            local_grad_diff_count += 1
        else:
            issues.append(
                f"step {s}: local_grad_sha1 IDENTICAL — gradients didn't "
                f"actually differ between ranks"
            )

        # Check 3: synced gradients match (allreduce averaged them)
        if a["synced_grad"]["sha1"] == b["synced_grad"]["sha1"]:
            synced_grad_match_count += 1
        else:
            issues.append(
                f"step {s}: synced_grad_sha1 DIVERGED — "
                f"rank0={a['synced_grad']['sha1'][:12]} "
                f"rank1={b['synced_grad']['sha1'][:12]} — sync broke"
            )

        # Check 4: params after step match
        if a["params_after_sha1"] == b["params_after_sha1"]:
            params_match_count += 1
        else:
            issues.append(
                f"step {s}: params_after DIVERGED — "
                f"rank0={a['params_after_sha1'][:12]} "
                f"rank1={b['params_after_sha1'][:12]}"
            )

    # Byte mirror checks
    bytes_sent_r0 = f0["total_bytes_sent"]
    bytes_recv_r0 = f0["total_bytes_received"]
    bytes_sent_r1 = f1["total_bytes_sent"]
    bytes_recv_r1 = f1["total_bytes_received"]

    # Allow 1% tolerance for handshake-level extras + counter timing
    def _mirror_ok(a: int, b: int) -> bool:
        if a == 0 and b == 0:
            return True
        return abs(a - b) / max(a, b) < 0.02

    sent_recv_ok = _mirror_ok(bytes_sent_r0, bytes_recv_r1)
    recv_sent_ok = _mirror_ok(bytes_recv_r0, bytes_sent_r1)

    # Final params hash
    final_match = f0["final_params_sha1"] == f1["final_params_sha1"]

    n = len(steps0)
    print("PER-STEP CHECKS")
    print(f"  batch_indices differ between ranks   {batch_diff_count}/{n}  "
          f"{color('PASS' if batch_diff_count == n else 'FAIL', batch_diff_count == n)}")
    print(f"  local_grad SHA1 differs between ranks {local_grad_diff_count}/{n}  "
          f"{color('PASS' if local_grad_diff_count == n else 'FAIL', local_grad_diff_count == n)}")
    print(f"  synced_grad SHA1 matches between ranks {synced_grad_match_count}/{n}  "
          f"{color('PASS' if synced_grad_match_count == n else 'FAIL', synced_grad_match_count == n)}")
    print(f"  params_after SHA1 matches between ranks {params_match_count}/{n}  "
          f"{color('PASS' if params_match_count == n else 'FAIL', params_match_count == n)}")
    print()

    print("BYTE MIRROR CHECKS")
    print(f"  rank 0 sent     = {bytes_sent_r0:>10}")
    print(f"  rank 1 received = {bytes_recv_r1:>10}  (delta={abs(bytes_sent_r0 - bytes_recv_r1)})  "
          f"{color('PASS' if sent_recv_ok else 'FAIL', sent_recv_ok)}")
    print(f"  rank 1 sent     = {bytes_sent_r1:>10}")
    print(f"  rank 0 received = {bytes_recv_r0:>10}  (delta={abs(bytes_sent_r1 - bytes_recv_r0)})  "
          f"{color('PASS' if recv_sent_ok else 'FAIL', recv_sent_ok)}")
    print()

    print("FINAL PARAMS")
    print(f"  rank 0 SHA1: {f0['final_params_sha1']}")
    print(f"  rank 1 SHA1: {f1['final_params_sha1']}")
    print(f"  match: {color('PASS' if final_match else 'FAIL', final_match)}")
    print()

    # Pretty-print first step as a sample
    print("=== STEP 0 SIDE-BY-SIDE ===")
    a0, a1 = steps0[0], steps1[0]
    print(f"  {'field':<25}  {'rank 0':<32}  {'rank 1':<32}")
    print(f"  {'-' * 25}  {'-' * 32}  {'-' * 32}")
    print(f"  {'batch_indices[:8]':<25}  "
          f"{str(a0['batch_indices'][:8]):<32}  {str(a1['batch_indices'][:8]):<32}")
    print(f"  {'loss':<25}  {a0['loss']:<32}  {a1['loss']:<32}")
    print(f"  {'local_grad first5':<25}  "
          f"{str(a0['local_grad']['first5']):<32}  {str(a1['local_grad']['first5']):<32}")
    print(f"  {'local_grad sha1':<25}  "
          f"{a0['local_grad']['sha1'][:24]:<32}  {a1['local_grad']['sha1'][:24]:<32}")
    print(f"  {'synced_grad first5':<25}  "
          f"{str(a0['synced_grad']['first5']):<32}  {str(a1['synced_grad']['first5']):<32}")
    print(f"  {'synced_grad sha1':<25}  "
          f"{a0['synced_grad']['sha1'][:24]:<32}  {a1['synced_grad']['sha1'][:24]:<32}")
    print(f"  {'params_after sha1':<25}  "
          f"{a0['params_after_sha1'][:24]:<32}  {a1['params_after_sha1'][:24]:<32}")
    print(f"  {'bytes_sent_delta':<25}  "
          f"{a0['bytes_sent_delta']:<32}  {a1['bytes_sent_delta']:<32}")
    print(f"  {'sync_ms':<25}  {a0['sync_ms']:<32}  {a1['sync_ms']:<32}")
    print()

    # Issues summary (first 10)
    if issues:
        print(f"=== ISSUES ({len(issues)} total, showing first 10) ===")
        for i in issues[:10]:
            print(f"  - {i}")
        print()

    # Overall verdict
    all_pass = (
        batch_diff_count == n and
        local_grad_diff_count == n and
        synced_grad_match_count == n and
        params_match_count == n and
        sent_recv_ok and recv_sent_ok and
        final_match
    )
    print("=" * 60)
    if all_pass:
        print(color("VERDICT: PASS — distributed training verified end-to-end.", True))
    else:
        print(color("VERDICT: FAIL — see issues above.", False))
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
