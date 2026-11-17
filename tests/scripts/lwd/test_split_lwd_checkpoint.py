# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for scripts/lwd/split_lwd_checkpoint.py (F2 checkpoint split).

Pinned behaviors (docs/design/lwd_weight_split.md section 7):

* rule coverage: Qwen3-VL / native DeepSeek layouts classify every tensor
  into exactly one side (or the split fails fast with exit code 3);
* mirror discipline: the vendored ownership snapshot equals the live
  ``LWD_MODULE_OWNERSHIP`` (AST check runs without vllm; an import check
  runs when vllm is installed) and keep/drop decisions equal the
  load-time filter for every non-special tensor name (the filter is
  loaded from source with an AST-seeded stub when vllm is absent);
* end-to-end: tensor sets, byte-identical payloads, ``__metadata__``
  preservation and per-role metadata rules on a synthetic checkpoint;
* engineering discipline: determinism, index rebuild around the
  single-shard companion trap, read-only guard, parallel writer.

The suite is stdlib + pytest only (the script itself has zero vllm
dependency), so offline hosts can run it with ``pytest --noconftest`` --
tests/conftest.py imports torch at module scope.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import struct
import sys
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "lwd" / "split_lwd_checkpoint.py"
UTILS_PATH = REPO_ROOT / "vllm" / "model_executor" / "models" / "utils.py"


def _load_script():
    """Import the stdlib-only script under its own module name.

    Registering the module in ``sys.modules`` keeps its functions picklable,
    which the ``--jobs 2`` parallel-writer path needs (ProcessPoolExecutor
    ships ``write_one_shard`` by reference).
    """
    spec = importlib.util.spec_from_file_location("split_lwd_checkpoint", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()

# ---------------------------------------------------------------------------
# Synthetic checkpoints (stdlib-only safetensors writer / reader)
# ---------------------------------------------------------------------------

BOTH = frozenset({"edge", "cloud"})
EDGE_ONLY = frozenset({"edge"})
CLOUD_ONLY = frozenset({"cloud"})

# Qwen3-VL-style layout: flat names, VL ``model.language_model.*`` nesting
# and a Hash MoE routing table under the cloud-owned layers subtree.
SHARD_TENSORS: dict[str, list[str]] = {
    "model-00001-of-00002.safetensors": [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.visual.blk.0.attn_qkv.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.ffn.gate.tid2eid",
        "lm_head.weight",
    ],
    "model-00002-of-00002.safetensors": [
        "model.language_model.model.embed_tokens.weight",
        "model.language_model.lm_head.weight",
        "model.language_model.model.norm.weight",
        "model.norm.weight",
        "model.mtp.proj.weight",
        # vllm-layout MTP spec layer (enorm + flat hc param) and the QuaRot
        # top-level draft input rotation.
        "model.layers.48.enorm.weight",
        "model.layers.48.hc_attn_fn",
        "rot.weight",
        "model.visual.merger.norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    ],
}

EXPECTED_OWNERSHIP: dict[str, frozenset[str]] = {
    "model.embed_tokens.weight": BOTH,
    "lm_head.weight": BOTH,
    "model.language_model.model.embed_tokens.weight": BOTH,
    "model.language_model.lm_head.weight": BOTH,
    "model.layers.0.ffn.gate.tid2eid": BOTH,  # edge: special case; cloud: layers
    "model.layers.0.self_attn.q_proj.weight": CLOUD_ONLY,
    "model.layers.0.mlp.down_proj.weight": CLOUD_ONLY,
    "model.language_model.model.norm.weight": CLOUD_ONLY,
    "model.norm.weight": CLOUD_ONLY,
    "model.mtp.proj.weight": CLOUD_ONLY,
    # MTP spec layer under the ``model.layers.{N}`` subtree (vllm layout) and
    # the QuaRot top-level draft input rotation (rides the MTP row).
    "model.layers.48.enorm.weight": CLOUD_ONLY,
    "model.layers.48.hc_attn_fn": CLOUD_ONLY,
    "rot.weight": CLOUD_ONLY,
    "model.visual.blk.0.attn_qkv.weight": EDGE_ONLY,
    "model.visual.merger.norm.weight": EDGE_ONLY,
    # DeepSeek-V4 flat hyper-connection head: both-roles special case
    "hc_head_fn": BOTH,
    "hc_head_base": BOTH,
    "hc_head_scale": BOTH,
}

# Native DeepSeek layout: no leading ``model.``, ``head`` / ``embed`` renames.
NATIVE_OWNERSHIP: dict[str, frozenset[str]] = {
    "embed_tokens.weight": BOTH,
    "embed.weight": BOTH,  # matched only via the embed alias row
    "head.weight": BOTH,  # matched only via the head alias row
    "layers.3.mlp.w1.weight": CLOUD_ONLY,
    "layers.5.ffn.gate.tid2eid.weight": BOTH,
    "norm.weight": CLOUD_ONLY,
    "mtp.0.linear.weight": CLOUD_ONLY,
    # QuaRot top-level draft input rotation: native layouts export it flat
    # too (the shape real GLM/DeepSeek w8a8c8 checkpoints ship).
    "rot.weight": CLOUD_ONLY,
    # DeepSeek-V4 flat hyper-connection head: both-roles special case
    "hc_head_fn": BOTH,
    "hc_head_base": BOTH,
    "hc_head_scale": BOTH,
}

# Kimi-K3 layout: VL nesting without the leading ``model.`` on the text
# tower, no ``visual`` prefix for the vision components, and the final norm's
# attention-residual tail as separate submodules. The four rows added for K3
# must classify every family below, i.e. the split may not leave any tensor
# uncovered (before the rows existed it aborted with RuleCoverageError).
K3_OWNERSHIP: dict[str, frozenset[str]] = {
    # vision side: edge-owned, and no ``model.`` prefix on the checkpoint path
    "vision_tower.encoder.blocks.0.attn.qkv.weight": EDGE_ONLY,
    # tower-internal norm must survive the model.norm rule
    "vision_tower.encoder.blocks.0.norm1.weight": EDGE_ONLY,
    "mm_projector.linear_1.weight": EDGE_ONLY,
    "mm_projector.proj.2.weight": EDGE_ONLY,
    # text side: the language model nests as language_model.model.*
    "language_model.model.embed_tokens.weight": BOTH,
    "language_model.lm_head.weight": BOTH,
    "language_model.model.layers.0.self_attn.q_proj.weight": CLOUD_ONLY,
    "language_model.model.norm.weight": CLOUD_ONLY,
    "language_model.model.output_attn_res_norm.weight": CLOUD_ONLY,
    "language_model.model.output_attn_res_proj.weight": CLOUD_ONLY,
}

K3_SHARD_TENSORS: dict[str, list[str]] = {
    "model-00001-of-00001.safetensors": list(K3_OWNERSHIP),
}

EDGE_METADATA_FILES = {
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "preprocessor_config.json",
    "chat_template.json",
}
CLOUD_METADATA_FILES = EDGE_METADATA_FILES - {"preprocessor_config.json"}


def _payload(name: str) -> bytes:
    """Deterministic per-name payload (unique bytes for every tensor)."""
    data = name.encode("utf-8")
    pad = 16 - len(data) % 16 or 16
    return (data + b"\x00" * pad) * 2


def _tensor_spec(name: str) -> tuple[str, list[int], bytes]:
    payload = _payload(name)
    return "F32", [len(payload) // 4], payload


def _write_safetensors(
    path: Path,
    tensors: Mapping[str, tuple[str, list[int], bytes]],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write a minimal safetensors file (u64 header length + JSON + data)."""
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body += b" " * ((-len(body)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
        for _, (_, _, payload) in tensors.items():
            handle.write(payload)


def _read_shard(path: Path) -> tuple[dict[str, bytes], dict | None]:
    """Read every tensor payload of a safetensors file (independent of the
    script's own header parser, so payload assertions are not tautological).
    """
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    metadata = header.pop("__metadata__", None)
    tensors: dict[str, bytes] = {}
    with open(path, "rb") as handle:
        base = 8 + header_len
        for name, info in sorted(header.items()):
            start, end = info["data_offsets"]
            handle.seek(base + start)
            tensors[name] = handle.read(end - start)
    return tensors, metadata


def _write_checkpoint(
    src: Path,
    *,
    shards: Mapping[str, Iterable[str]] | None = None,
    index_name: str | None = "model.safetensors.index.json",
    companions: Mapping[str, Mapping[str, tuple[str, list[int], bytes]]] | None = None,
) -> None:
    """Write a complete checkpoint: shards, optional index, companions."""
    shard_map = dict(SHARD_TENSORS if shards is None else shards)
    weight_map: dict[str, str] = {}
    total = 0
    for filename, names in sorted(shard_map.items()):
        _write_safetensors(
            src / filename,
            {name: _tensor_spec(name) for name in names},
            metadata={"format": "pt"},
        )
        for name in names:
            weight_map[name] = filename
            total += len(_payload(name))
    for filename, tensors in sorted((companions or {}).items()):
        _write_safetensors(src / filename, tensors)
    if index_name is not None:
        payload = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (src / index_name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def _write_standard_metadata(src: Path) -> None:
    """Files every HF checkpoint dir has; the roles differ on them."""
    for name in sorted(EDGE_METADATA_FILES | {"README.md"}):
        (src / name).write_text("{}\n", encoding="utf-8")
    (src / "optional").mkdir()
    (src / "optional" / "rotation_meta.json").write_text("{}\n", encoding="utf-8")
    (src / "extra_dir").mkdir()
    (src / "extra_dir" / "notes.txt").write_text("keep me out\n", encoding="utf-8")


def _split(src: Path, dst: Path, role: str, *extra: str) -> int:
    return script.main(
        ["--src", str(src), "--dst", str(dst), "--role", role, "--no-progress", *extra]
    )


def _output_shards(directory: Path) -> list[Path]:
    return sorted(directory.glob("model-*-of-*.safetensors"))


def _output_tensors(
    directory: Path,
) -> tuple[dict[str, bytes], dict[str, dict | None]]:
    tensors: dict[str, bytes] = {}
    metadatas: dict[str, dict | None] = {}
    for shard in _output_shards(directory):
        shard_tensors, metadata = _read_shard(shard)
        tensors.update(shard_tensors)
        metadatas[shard.name] = metadata
    return tensors, metadatas


def _source_tensors(src: Path) -> dict[str, bytes]:
    tensors: dict[str, bytes] = {}
    for filename in SHARD_TENSORS:
        shard_tensors, _ = _read_shard(src / filename)
        tensors.update(shard_tensors)
    return tensors


def _expected_keep(role: str) -> set[str]:
    return {name for name, owners in EXPECTED_OWNERSHIP.items() if role in owners}


# ---------------------------------------------------------------------------
# Rule coverage (design section 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_classify_covers_qwen3_vl_layout(role: str) -> None:
    table, _ = script.load_module_ownership()
    weight_map = {name: "x.safetensors" for name in EXPECTED_OWNERSHIP}
    keep, drop, uncovered, _ = script.classify_all(weight_map, role, table)
    expected_keep = _expected_keep(role)
    assert uncovered == []
    assert keep == expected_keep
    assert drop == set(weight_map) - expected_keep
    assert not keep & drop


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_classify_covers_native_deepseek_layout(role: str) -> None:
    table, _ = script.load_module_ownership()
    weight_map = {name: "x.safetensors" for name in NATIVE_OWNERSHIP}
    keep, drop, uncovered, _ = script.classify_all(weight_map, role, table)
    expected_keep = {n for n, owners in NATIVE_OWNERSHIP.items() if role in owners}
    assert uncovered == []
    assert keep == expected_keep
    assert drop == set(weight_map) - expected_keep


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_classify_covers_kimi_k3_layout(role: str) -> None:
    """Every Kimi-K3 tensor family is owned; none falls into the fail-fast bucket."""
    table, _ = script.load_module_ownership()
    weight_map = {name: "x.safetensors" for name in K3_OWNERSHIP}
    keep, drop, uncovered, _ = script.classify_all(weight_map, role, table)
    expected_keep = {n for n, owners in K3_OWNERSHIP.items() if role in owners}
    assert uncovered == []
    assert keep == expected_keep
    assert drop == set(weight_map) - expected_keep
    assert not keep & drop


def test_kimi_k3_layout_is_uncovered_without_the_new_rows() -> None:
    """Pins the failure mode the four K3 rows fix.

    Dropping any of them puts the matching tensors back into the uncovered
    bucket, which makes the split abort with RuleCoverageError (exit 3) --
    i.e. the K3 checkpoint could not be split at all.
    """
    table, _ = script.load_module_ownership()
    k3_rows = {
        "vision_tower",
        "mm_projector",
        "output_attn_res_norm",
        "output_attn_res_proj",
    }
    assert k3_rows <= table.keys()
    legacy = {k: v for k, v in table.items() if k not in k3_rows}
    weight_map = {name: "x.safetensors" for name in K3_OWNERSHIP}
    _, _, uncovered, _ = script.classify_all(weight_map, "edge", legacy)
    assert {
        "vision_tower.encoder.blocks.0.attn.qkv.weight",
        "mm_projector.linear_1.weight",
        "language_model.model.output_attn_res_norm.weight",
        "language_model.model.output_attn_res_proj.weight",
    } <= set(uncovered)


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_quarot_mtp_rot_tensor_belongs_to_cloud(role: str) -> None:
    """The QuaRot top-level ``rot.weight`` rides the MTP row in every layout.

    QuaRot-quantized MTP checkpoints export the draft input rotation as a
    flat top-level tensor (no ``model.`` prefix, in flat and native layouts
    alike). The cloud draft consumes it (remapped into the spec layer); the
    edge target discards ``rot.*`` via its weights mapper, so the edge slice
    omits it.
    """
    table, _ = script.load_module_ownership()
    weight_map = {"rot.weight": "x.safetensors"}
    keep, drop, uncovered, attribution = script.classify_all(weight_map, role, table)
    assert uncovered == []
    assert attribution["table:model.mtp"] == ["rot.weight"]
    if role == "cloud":
        assert keep == {"rot.weight"}
        assert drop == set()
    else:
        assert keep == set()
        assert drop == {"rot.weight"}


def test_keep_iff_all_matched_rows_owned() -> None:
    """A tensor matching two rows is kept only when the role owns every row.

    This pins the ALL semantics shared with the load-time filter: the name
    below matches ``row_a`` (owned by both) and ``row_b`` (cloud only), so
    edge must drop it while cloud keeps it.
    """
    table = {
        "row_a": frozenset({"edge", "cloud"}),
        "row_b": frozenset({"cloud"}),
    }
    weight_map = {"row_a.row_b.weight": "x.safetensors"}
    edge_keep, edge_drop, uncovered, _ = script.classify_all(weight_map, "edge", table)
    cloud_keep, cloud_drop, _, _ = script.classify_all(weight_map, "cloud", table)
    assert uncovered == []
    assert edge_keep == set()
    assert edge_drop == {"row_a.row_b.weight"}
    assert cloud_keep == {"row_a.row_b.weight"}
    assert cloud_drop == set()


# ---------------------------------------------------------------------------
# Mirror discipline: vendored table vs live table vs load-time filter
# ---------------------------------------------------------------------------


def _eval_frozenset_expr(
    node: ast.expr, env: dict[str, frozenset[str]]
) -> frozenset[str]:
    """Evaluate ``frozenset(("edge", ...))`` calls and name references."""
    if isinstance(node, ast.Name):
        return env[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.Tuple, ast.List))
    ):
        values = []
        for element in node.args[0].elts:
            if not (
                isinstance(element, ast.Constant) and isinstance(element.value, str)
            ):
                raise ValueError(f"unsupported element: {ast.dump(element)}")
            values.append(element.value)
        return frozenset(values)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _parse_live_ownership() -> dict[str, frozenset[str]]:
    """Extract LWD_MODULE_OWNERSHIP from models/utils.py without importing
    vllm (the AST walk resolves the ``_LWD_BOTH`` indirection)."""
    tree = ast.parse(UTILS_PATH.read_text(encoding="utf-8"))
    env: dict[str, frozenset[str]] = {}
    table: dict[str, frozenset[str]] | None = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is None:
                continue
            assignments = [(node.target.id, node.value)]
        elif isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) for target in node.targets
        ):
            assignments = [(target.id, node.value) for target in node.targets]  # type: ignore[attr-defined]
        else:
            continue
        for name, value in assignments:
            if name == "_LWD_BOTH":
                env[name] = _eval_frozenset_expr(value, env)
            elif name == "LWD_MODULE_OWNERSHIP" and isinstance(value, ast.Dict):
                table = {
                    key.value: _eval_frozenset_expr(val, env)
                    for key, val in zip(value.keys, value.values)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
    assert table is not None, "LWD_MODULE_OWNERSHIP not found in models/utils.py"
    return table


def test_vendored_ownership_table_equals_live_table() -> None:
    """The vendored snapshot must never drift from F1's table."""
    assert _parse_live_ownership() == script._VENDORED_MODULE_OWNERSHIP


def _load_weight_filter():
    """Import the load-time filter, without torch when necessary.

    ``lwd_weight_filter`` only needs ``LWD_MODULE_OWNERSHIP`` from
    ``vllm.model_executor.models.utils``; on hosts without a vllm install
    the module is loaded from source with a stub for that import, seeded
    from the AST-parsed live table (the real table, not the vendored one,
    so the mirror test below never compares two copies of the snapshot).
    """
    try:
        return importlib.import_module(
            "vllm.model_executor.model_loader.lwd_weight_filter"
        )
    except ImportError:
        pass
    stub_name = "vllm.model_executor.models.utils"
    filter_name = "vllm.model_executor.model_loader.lwd_weight_filter"
    utils_stub: Any = types.ModuleType(stub_name)
    utils_stub.LWD_MODULE_OWNERSHIP = _parse_live_ownership()
    utils_stub.LwdRole = str  # typing-only alias in the source module
    sys.modules[stub_name] = utils_stub
    try:
        filter_path = (
            REPO_ROOT
            / "vllm"
            / "model_executor"
            / "model_loader"
            / "lwd_weight_filter.py"
        )
        spec = importlib.util.spec_from_file_location(filter_name, filter_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[filter_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(stub_name, None)
        sys.modules.pop(filter_name, None)


def test_vendored_ownership_table_equals_live_import() -> None:
    """Same equality against the imported table (vllm environments only)."""
    utils = pytest.importorskip("vllm.model_executor.models.utils")
    live = {str(k): frozenset(v) for k, v in utils.LWD_MODULE_OWNERSHIP.items()}
    assert live == script._VENDORED_MODULE_OWNERSHIP


def test_load_module_ownership_prefers_live() -> None:
    pytest.importorskip("vllm.model_executor.models.utils")
    table, live = script.load_module_ownership()
    assert live is True
    assert table == _parse_live_ownership()


def test_classification_mirrors_load_time_filter() -> None:
    """Split drop == load-time skip for every name except the tid2eid edge
    keep (vllm-ascend reads it by path from the edge shards)."""
    weight_filter = _load_weight_filter()
    table, _ = script.load_module_ownership()
    names = sorted(EXPECTED_OWNERSHIP | NATIVE_OWNERSHIP)
    for role in ("edge", "cloud"):
        weight_map = {name: "x.safetensors" for name in names}
        keep, drop, uncovered, _ = script.classify_all(weight_map, role, table)
        assert uncovered == []
        for name in names:
            if ".tid2eid" in name:
                assert name in keep
                if role == "edge":
                    assert weight_filter.lwd_should_skip_weight(name, role)
                continue
            if name == "rot.weight":
                # QuaRot draft input rotation: the split routes it with the
                # MTP row (edge drops it from the slice), while the load-time
                # filter passes it through on both roles -- the edge target
                # discards ``rot.*`` via its weights mapper and the cloud
                # draft remaps it into the spec layer. Asymmetry by design,
                # the same shape as the tid2eid exemption above.
                if role == "edge":
                    assert name in drop
                else:
                    assert name in keep
                assert not weight_filter.lwd_should_skip_weight(name, role)
                continue
            should_skip = weight_filter.lwd_should_skip_weight(name, role)
            assert (name in drop) == should_skip, (role, name)
            assert (name in keep) == (not should_skip), (role, name)


# ---------------------------------------------------------------------------
# End-to-end split
# ---------------------------------------------------------------------------


def test_end_to_end_edge_split(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src)
    _write_standard_metadata(src)
    dst = tmp_path / "edge"

    assert _split(src, dst, "edge") == 0

    tensors, metadatas = _output_tensors(dst)
    assert set(tensors) == _expected_keep("edge")
    assert tensors == {
        name: payload
        for name, payload in _source_tensors(src).items()
        if name in tensors
    }
    assert metadatas and all(m == {"format": "pt"} for m in metadatas.values())

    # edge is the serving front-end: processor + tokenizer files, the
    # QuaRot optional/ dir, no docs noise, no index (single shard).
    entries = {entry.name for entry in dst.iterdir()}
    shards = {path.name for path in _output_shards(dst)}
    assert entries == shards | EDGE_METADATA_FILES | {"optional"}
    assert not (dst / "model.safetensors.index.json").exists()


def test_end_to_end_cloud_split(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src)
    _write_standard_metadata(src)
    dst = tmp_path / "cloud"

    assert _split(src, dst, "cloud") == 0

    tensors, _ = _output_tensors(dst)
    assert set(tensors) == _expected_keep("cloud")
    assert tensors == {
        name: payload
        for name, payload in _source_tensors(src).items()
        if name in tensors
    }

    # cloud is a pure decoding engine: processor files stay behind.
    entries = {entry.name for entry in dst.iterdir()}
    shards = {path.name for path in _output_shards(dst)}
    assert entries == shards | CLOUD_METADATA_FILES | {"optional"}


def _write_k3_checkpoint(src: Path) -> None:
    _write_checkpoint(src, shards=K3_SHARD_TENSORS)


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_kimi_k3_split_end_to_end(tmp_path: Path, role: str) -> None:
    """A K3-shaped checkpoint splits cleanly and keeps the right tensors.

    Regression for the LWD split: before the K3 rows existed this aborted with
    RuleCoverageError (uncovered vision_tower / mm_projector / attn-res tails).
    """
    src = tmp_path / "src"
    src.mkdir()
    _write_k3_checkpoint(src)
    _write_standard_metadata(src)
    dst = tmp_path / role

    assert _split(src, dst, role) == 0

    tensors, metadatas = _output_tensors(dst)
    expected = {n for n, owners in K3_OWNERSHIP.items() if role in owners}
    assert set(tensors) == expected
    assert tensors == {name: _payload(name) for name in expected}
    assert metadatas and all(m == {"format": "pt"} for m in metadatas.values())


def test_split_is_deterministic(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src)
    _write_standard_metadata(src)
    first, second = tmp_path / "a", tmp_path / "b"
    # 1e-08 GiB caps every shard below one tensor payload: multi-shard + index
    extra = ("--max-shard-gb", "1e-08")

    assert _split(src, first, "edge", *extra) == 0
    assert _split(src, second, "edge", *extra) == 0
    assert len(_output_shards(first)) > 1

    names_first = sorted(
        str(p.relative_to(first)) for p in first.rglob("*") if p.is_file()
    )
    names_second = sorted(
        str(p.relative_to(second)) for p in second.rglob("*") if p.is_file()
    )
    assert names_first == names_second
    for rel in names_first:
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), rel


def test_metadata_all_copies_everything(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src)
    _write_standard_metadata(src)
    dst = tmp_path / "cloud"

    assert _split(src, dst, "cloud", "--metadata", "all") == 0

    entries = {entry.name for entry in dst.iterdir()}
    # preprocessor files and unlisted directories ship only with --metadata
    # all; the skip rules (README etc.) stay in effect either way.
    assert "preprocessor_config.json" in entries
    assert "extra_dir" in entries
    assert (dst / "extra_dir" / "notes.txt").is_file()
    assert "README.md" not in entries


def test_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src)
    _write_standard_metadata(src)
    before = sorted(entry.name for entry in src.iterdir())

    rc = script.main(
        ["--src", str(src), "--role", "edge", "--dry-run", "--no-progress"]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "[plan]" in captured.out
    assert "dry-run" in captured.out
    assert sorted(entry.name for entry in src.iterdir()) == before


# ---------------------------------------------------------------------------
# Failure branches
# ---------------------------------------------------------------------------


def test_uncovered_tensor_fails_fast(tmp_path: Path, capsys) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(
        src,
        shards={
            "model-00001-of-00001.safetensors": [
                "marker_head.weight",
                "model.norm.weight",
            ]
        },
    )
    dst = tmp_path / "out"

    # marker_head matches no ownership row and no special case: refuse to
    # guess instead of silently dropping it.
    assert _split(src, dst, "cloud") == 3
    captured = capsys.readouterr()
    assert "marker_head" in captured.out + captured.err
    assert not dst.exists()

    # explicit operator classification lets the split proceed
    assert _split(src, dst, "cloud", "--keep-prefix", "marker_head") == 0
    assert (
        _split(
            src, tmp_path / "out-regex", "cloud", "--keep-name-regex", r"^marker_head\."
        )
        == 0
    )


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_deepseek_v4_hc_head_kept_for_both_roles(
    role: str, tmp_path: Path, capsys
) -> None:
    """Regression: DeepSeek-V4-Flash ships ``hc_head_{fn,base,scale}`` as FLAT
    parameters (no dot boundary), so no ownership row can match them and the
    coverage check failed with exit code 3. They must classify without CLI
    flags: both roles keep them (the load-time filter skips them for neither
    and both roles construct the parameters; only the cloud runs the kernel).
    """
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(
        src,
        shards={
            "model-00001-of-00001.safetensors": [
                "hc_head_fn",
                "hc_head_base",
                "hc_head_scale",
                "model.norm.weight",
                "lm_head.weight",
            ]
        },
    )

    # dry-run classifies everything without CLI flags
    rc = script.main(["--src", str(src), "--role", role, "--dry-run", "--no-progress"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[plan] special:hc_head: 3 tensors" in captured.out
    assert "matched no rule" not in captured.out

    # a real split keeps the three tensors in the role's output
    dst = tmp_path / "out"
    assert _split(src, dst, role) == 0
    tensors, _ = _output_tensors(dst)
    assert {"hc_head_fn", "hc_head_base", "hc_head_scale"} <= set(tensors)
    assert tensors["hc_head_fn"] == _payload("hc_head_fn")


def test_usage_errors(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src)
    base = ["--src", str(src), "--role", "edge", "--no-progress"]

    assert script.main([*base, "--dst", str(src)]) == 2  # dst == src
    assert script.main([*base, "--dst", str(src / "nested")]) == 2  # inside src
    assert not (src / "nested").exists()
    assert script.main(base) == 2  # --dst required without --dry-run
    assert script.main([*base, "--keep-name-regex", "["]) == 2  # invalid regex


def test_readonly_guard_aborts_on_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src)
    victim = src / "model-00001-of-00002.safetensors"
    original = script.write_one_shard

    def touching_writer(shard, *args, **kwargs):
        original(shard, *args, **kwargs)
        stat = os.stat(victim)
        os.utime(victim, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    monkeypatch.setattr(script, "write_one_shard", touching_writer)
    assert _split(src, tmp_path / "guarded", "edge", "--jobs", "1") == 6
    assert (
        _split(
            src,
            tmp_path / "unguarded",
            "edge",
            "--jobs",
            "1",
            "--no-verify-src-readonly",
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Index handling
# ---------------------------------------------------------------------------


def test_index_rebuilt_for_single_shard_with_companion(tmp_path: Path) -> None:
    """The trap: a single-shard output normally needs no index, but a
    copied companion safetensors makes one mandatory."""
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(
        src,
        companions={
            "rot.safetensors": {
                "model.layers.99.self_attn.q_proj.weight": _tensor_spec(
                    "model.layers.99.self_attn.q_proj.weight"
                )
            }
        },
    )

    for role in ("edge", "cloud"):
        dst = tmp_path / f"out-{role}"
        assert _split(src, dst, role) == 0

        index_path = dst / "model.safetensors.index.json"
        assert index_path.is_file()
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        assert set(weight_map) == _expected_keep(role)
        assert set(weight_map.values()) == {"model-00001-of-00001.safetensors"}
        # the companion is shipped verbatim by both roles
        assert (dst / "rot.safetensors").read_bytes() == (
            src / "rot.safetensors"
        ).read_bytes()


def test_custom_index_name_is_never_copied(tmp_path: Path) -> None:
    """Sources may use e.g. quant_model_weights.safetensors.index.json; the
    output always rebuilds its own index, so the stale name must not leak
    into the output next to it."""
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src, index_name="quant_model_weights.safetensors.index.json")
    dst = tmp_path / "out"

    assert _split(src, dst, "edge", "--max-shard-gb", "1e-08") == 0

    entries = {entry.name for entry in dst.iterdir()}
    assert len(_output_shards(dst)) > 1  # multi-shard: the index is required
    assert "quant_model_weights.safetensors.index.json" not in entries
    assert "model.safetensors.index.json" in entries
    weight_map = json.loads(
        (dst / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    assert set(weight_map) == _expected_keep("edge")
    assert set(weight_map.values()) <= entries


def test_no_index_fallback_skips_companion_tensors(tmp_path: Path) -> None:
    """Without an index the shards are enumerated from headers; companion
    safetensors must stay out of that weight map (they are read by path at
    runtime, and their tensors would be missing from the weight shards)."""
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(
        src,
        index_name=None,
        companions={
            "rot.safetensors": {
                "model.layers.99.rot.weight": _tensor_spec("model.layers.99.rot.weight")
            }
        },
    )
    dst = tmp_path / "out"

    assert _split(src, dst, "cloud") == 0

    tensors, _ = _output_tensors(dst)
    assert set(tensors) == _expected_keep("cloud")
    assert "model.layers.99.rot.weight" not in tensors
    assert (dst / "rot.safetensors").is_file()
    # single shard + companion -> the rebuilt index is mandatory here too
    assert (dst / "model.safetensors.index.json").is_file()


# ---------------------------------------------------------------------------
# Parallel writer
# ---------------------------------------------------------------------------


def test_parallel_writer_matches_serial(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src)
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    extra = ("--max-shard-gb", "1e-08")  # multiple shards -> real pool path

    assert _split(src, serial, "edge", "--jobs", "1", *extra) == 0
    assert _split(src, parallel, "edge", "--jobs", "2", *extra) == 0
    assert len(_output_shards(parallel)) > 1

    serial_tensors, _ = _output_tensors(serial)
    parallel_tensors, _ = _output_tensors(parallel)
    assert (
        serial_tensors
        == parallel_tensors
        == {
            name: payload
            for name, payload in _source_tensors(src).items()
            if name in serial_tensors
        }
    )
