# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the LWD edge/cloud top-20 parity checker.

The checker is the machine-readable half of the MR4 parity smoke: the edge
and the cloud each log one ``[layer-trace] ... top20=`` line per replayed
batch, and this tool decides whether the two sides agree. The tests build
synthetic log lines, so no NPU is needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "lwd" / "lwd_trace_parity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("lwd_trace_parity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _line(side: str, ids: list[int], *, l2: float = 1.0) -> str:
    top = [(token, round(10.0 - index, 3)) for index, token in enumerate(ids)]
    return (
        "2026-11-17 10:00:00,000 INFO [layer-trace] "
        f"{side} lm_head logits shape=(1, 32000) row0 l2={l2:.4f} top20={top}"
    )


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ids(start: int = 100) -> list[int]:
    return list(range(start, start + 20))


def test_parse_trace_lines_extracts_both_sides() -> None:
    text = "\n".join(
        [
            "noise line without a trace",
            _line("edge", _ids()),
            _line("cloud", _ids(), l2=2.5),
            "[layer-trace] cloud ==== forward step=1 ====",
        ]
    )
    records = script.parse_trace_lines(text)
    assert [record.side for record in records] == ["edge", "cloud"]
    assert records[0].top1 == 100
    assert records[0].ids == _ids()
    assert records[1].l2 == pytest.approx(2.5)


def test_parse_trace_lines_filters_by_side() -> None:
    text = "\n".join([_line("edge", _ids()), _line("cloud", _ids())])
    assert [r.side for r in script.parse_trace_lines(text, side="edge")] == ["edge"]
    assert [r.side for r in script.parse_trace_lines(text, side="cloud")] == ["cloud"]


def test_identical_logs_pass(tmp_path: Path) -> None:
    edge = _write(tmp_path / "edge.log", [_line("edge", _ids())])
    cloud = _write(tmp_path / "cloud.log", [_line("cloud", _ids())])
    code, report = script.run(edge, cloud, min_overlap=18)
    assert code == 0
    assert "verdict: PASS" in report
    assert "| 0 | 100 | 100 | 20/20 |" in report


def test_top1_swap_fails(tmp_path: Path) -> None:
    edge = _write(tmp_path / "edge.log", [_line("edge", _ids(100))])
    cloud_ids = _ids(100)
    cloud_ids[0], cloud_ids[1] = cloud_ids[1], cloud_ids[0]
    cloud = _write(tmp_path / "cloud.log", [_line("cloud", cloud_ids)])
    code, report = script.run(edge, cloud, min_overlap=18)
    assert code == 1
    assert "top-1 100 != 101" in report


def test_low_overlap_fails(tmp_path: Path) -> None:
    edge = _write(tmp_path / "edge.log", [_line("edge", _ids(100))])
    cloud = _write(tmp_path / "cloud.log", [_line("cloud", _ids(500))])
    code, report = script.run(edge, cloud, min_overlap=18)
    assert code == 1
    assert "overlap 0/20 < 18" in report


def test_min_overlap_threshold_is_enforced(tmp_path: Path) -> None:
    edge_ids = _ids(100)
    cloud_ids = edge_ids[:-1] + [999]  # 19 shared tokens
    edge = _write(tmp_path / "edge.log", [_line("edge", edge_ids)])
    cloud = _write(tmp_path / "cloud.log", [_line("cloud", cloud_ids)])
    assert script.run(edge, cloud, min_overlap=19)[0] == 0
    code, report = script.run(edge, cloud, min_overlap=20)
    assert code == 1
    assert "overlap 19/20 < 20" in report


def test_missing_side_reports_insufficient_data(tmp_path: Path) -> None:
    edge = _write(tmp_path / "edge.log", [_line("edge", _ids())])
    cloud = _write(tmp_path / "cloud.log", ["no traces here"])
    code, report = script.run(edge, cloud, min_overlap=18)
    assert code == 2
    assert "INSUFFICIENT DATA" in report
    assert "no cloud trace lines" in report


def test_batch_count_mismatch_fails(tmp_path: Path) -> None:
    edge = _write(
        tmp_path / "edge.log", [_line("edge", _ids()), _line("edge", _ids(200))]
    )
    cloud = _write(tmp_path / "cloud.log", [_line("cloud", _ids())])
    code, report = script.run(edge, cloud, min_overlap=18)
    assert code == 1
    assert "batch counts differ" in report
    assert "paired batches: 1" in report
    # The warning must carry the interpolated count, not a literal brace
    # (regression: the second half of the warning line missed its f-prefix).
    assert "only the first 1 batches were compared" in report


def test_cli_writes_report(tmp_path: Path) -> None:
    edge = _write(tmp_path / "edge.log", [_line("edge", _ids())])
    cloud = _write(tmp_path / "cloud.log", [_line("cloud", _ids())])
    report_path = tmp_path / "report.md"
    code = script.main(
        [
            "--edge",
            str(edge),
            "--cloud",
            str(cloud),
            "--min-overlap",
            "20",
            "--report",
            str(report_path),
        ]
    )
    assert code == 0
    assert report_path.read_text(encoding="utf-8").startswith(
        "# LWD edge/cloud top-20 parity report"
    )


def test_cli_rejects_out_of_range_min_overlap(tmp_path: Path) -> None:
    edge = _write(tmp_path / "edge.log", [_line("edge", _ids())])
    cloud = _write(tmp_path / "cloud.log", [_line("cloud", _ids())])
    with pytest.raises(SystemExit):
        script.main(
            [
                "--edge",
                str(edge),
                "--cloud",
                str(cloud),
                "--min-overlap",
                "21",
            ]
        )
