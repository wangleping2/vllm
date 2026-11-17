#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the LWD edge/cloud ``[layer-trace]`` top-20 lines (L0 parity smoke).

Both sides log one line per replayed batch while
``VLLM_ASCEND_LWD_LAYER_TRACE=1``:

    [layer-trace] <side> lm_head logits shape=(R, V) row0 l2=... top20=[(id, val), ...]

The edge computes those logits from the DOWN payload, the cloud from its own
sampler, so the two lines are directly comparable: a rank swap or a drifted
top-1 is replay divergence, not noise. The lines carry no sequence number, so
they are paired in log order -- meaningful for a single-connection parity run
(one prompt, text only), which is exactly what the smoke does.

Exit codes:
    0  every paired batch agrees (top-1 equal, overlap >= --min-overlap)
    1  at least one batch disagrees
    2  insufficient data (a side has no trace lines, or the batch counts differ)

Usage:
    python scripts/lwd/lwd_trace_parity.py --edge edge.log --cloud cloud.log
    python scripts/lwd/lwd_trace_parity.py ... --report parity.md --min-overlap 20
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TOP = 20
_PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)")
_TRACE_RE = re.compile(r"\[layer-trace\]\s+(edge|cloud)\s+lm_head\s+logits\b")
_L2_RE = re.compile(r"row0\s+l2=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


@dataclass(frozen=True)
class TraceRecord:
    """One side's top-20 for a single replayed batch."""

    side: str
    l2: float
    top: tuple[tuple[int, float], ...]

    @property
    def top1(self) -> int:
        return self.top[0][0] if self.top else -1

    @property
    def ids(self) -> list[int]:
        return [token for token, _ in self.top]


@dataclass(frozen=True)
class Pair:
    """One paired batch and its agreement metrics."""

    index: int
    edge: TraceRecord
    cloud: TraceRecord

    @property
    def overlap(self) -> int:
        return len(set(self.edge.ids) & set(self.cloud.ids))

    @property
    def top1_equal(self) -> bool:
        return self.edge.top1 == self.cloud.top1

    @property
    def same_rank(self) -> bool:
        return self.edge.ids[:TOP] == self.cloud.ids[:TOP]


def parse_trace_lines(text: str, *, side: str | None = None) -> list[TraceRecord]:
    """Extract one side's trace records from a log, in file order."""
    records: list[TraceRecord] = []
    for line in text.splitlines():
        match = _TRACE_RE.search(line)
        if match is None or (side is not None and match.group(1) != side):
            continue
        tail = line.split("top20=", 1)
        if len(tail) != 2:
            continue
        pairs = tuple(
            (int(token), float(value)) for token, value in _PAIR_RE.findall(tail[1])
        )
        if not pairs:
            continue
        l2_match = _L2_RE.search(line)
        records.append(
            TraceRecord(
                side=match.group(1),
                l2=float(l2_match.group(1)) if l2_match else float("nan"),
                top=pairs,
            )
        )
    return records


def pair_records(
    edge: Sequence[TraceRecord], cloud: Sequence[TraceRecord]
) -> list[Pair]:
    """Pair the two sides in log order (see the module docstring)."""
    return [
        Pair(index=index, edge=edge_record, cloud=cloud_record)
        for index, (edge_record, cloud_record) in enumerate(zip(edge, cloud))
    ]


def verdict(pairs: Sequence[Pair], *, min_overlap: int) -> tuple[bool, list[str]]:
    """Return (passed, human-readable failure reasons)."""
    reasons: list[str] = []
    for pair in pairs:
        if not pair.top1_equal:
            reasons.append(
                f"batch {pair.index}: top-1 {pair.edge.top1} != {pair.cloud.top1}"
            )
        if pair.overlap < min_overlap:
            reasons.append(
                f"batch {pair.index}: top-{TOP} overlap {pair.overlap}/"
                f"{TOP} < {min_overlap}"
            )
    return (not reasons), reasons


def format_report(
    pairs: Sequence[Pair],
    *,
    edge_total: int,
    cloud_total: int,
    min_overlap: int,
    passed: bool,
    reasons: Sequence[str],
) -> str:
    """Render a compact Markdown parity report."""
    lines = [
        "# LWD edge/cloud top-20 parity report",
        "",
        f"- edge trace lines: {edge_total}",
        f"- cloud trace lines: {cloud_total}",
        f"- paired batches: {len(pairs)}",
        f"- required top-{TOP} overlap: {min_overlap}/{TOP}",
        f"- verdict: {'PASS' if passed else 'FAIL'}",
    ]
    if edge_total != cloud_total:
        lines.append(
            f"- warning: batch counts differ (edge {edge_total}, cloud {cloud_total});"
            f" only the first {len(pairs)} batches were compared"
        )
    lines += [
        "",
        f"| batch | edge top1 | cloud top1 | top-{TOP} overlap | edge l2 | cloud l2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair.index} | {pair.edge.top1} | {pair.cloud.top1} |"
            f" {pair.overlap}/{TOP} | {pair.edge.l2:.4f} | {pair.cloud.l2:.4f} |"
        )
    if reasons:
        lines += ["", "## Failures", ""]
        lines += [f"- {reason}" for reason in reasons]
    return "\n".join(lines) + "\n"


def run(edge_path: Path, cloud_path: Path, *, min_overlap: int) -> tuple[int, str]:
    """Compare two logs; return (exit code, report text)."""
    edge = parse_trace_lines(edge_path.read_text(encoding="utf-8"), side="edge")
    cloud = parse_trace_lines(cloud_path.read_text(encoding="utf-8"), side="cloud")
    if not edge or not cloud:
        missing = "edge" if not edge else "cloud"
        report = (
            "# LWD edge/cloud top-20 parity report\n\n"
            f"- edge trace lines: {len(edge)}\n"
            f"- cloud trace lines: {len(cloud)}\n"
            f"- verdict: INSUFFICIENT DATA (no {missing} trace lines;"
            " enable VLLM_ASCEND_LWD_LAYER_TRACE=1 on both sides)\n"
        )
        return 2, report
    pairs = pair_records(edge, cloud)
    passed, reasons = verdict(pairs, min_overlap=min_overlap)
    report = format_report(
        pairs,
        edge_total=len(edge),
        cloud_total=len(cloud),
        min_overlap=min_overlap,
        passed=passed,
        reasons=reasons,
    )
    if passed and len(edge) == len(cloud):
        return 0, report
    return 1, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edge", type=Path, required=True, help="edge process log")
    parser.add_argument("--cloud", type=Path, required=True, help="cloud process log")
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=TOP - 2,
        help=f"required token overlap in the top-{TOP} (default: {TOP - 2})",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="write the Markdown report here"
    )
    args = parser.parse_args(argv)

    if not 0 <= args.min_overlap <= TOP:
        parser.error(f"--min-overlap must be within [0, {TOP}]")

    code, report = run(args.edge, args.cloud, min_overlap=args.min_overlap)
    if args.report is not None:
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
