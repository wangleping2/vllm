#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split a complete HF safetensors checkpoint into per-role LWD directories.

LWD runs one model in two cooperating processes (see
``docs/design/lwd_migration_overview.md``): the edge keeps embeddings /
lm_head / the vision tower and serves the multimodal front-end, the cloud
owns the decoder layers, the final norm and the MTP weights. F1 already
prunes the unowned modules at build and load time; this script additionally
splits the checkpoint *on disk* so each side only ships its own weights.

Why tensor level: shards of a single checkpoint interleave tensor names, so
file-level copy cannot produce clean role directories. The script streams
each source shard once, writes new shards containing only the tensors the
role owns, and rebuilds the index.

Rules (mirror discipline, ``docs/design/lwd_weight_split.md``):

* Keep/drop decisions derive from ``LWD_MODULE_OWNERSHIP`` in
  ``vllm/model_executor/models/utils.py`` (F1) -- the same single source of
  truth the load-time filter uses. On hosts without a vllm install the
  script falls back to a vendored snapshot of the table;
  ``tests/scripts/lwd/test_split_lwd_checkpoint.py`` asserts the two never
  drift.
* Native DeepSeek layout aliases (``layers.`` / ``embed.`` / ``head.`` /
  ``norm.`` / ``mtp.`` without the leading ``model.``) are explicitly
  registered per ownership row.
* Kimi-K3/K2.5 layouts name the vision tower and projector
  ``vision_tower.*`` / ``mm_projector.*`` (no ``model.`` prefix) and keep the
  final-norm attention-residual tail under
  ``language_model.model.output_attn_res_{norm,proj}.*``. All four families
  are owned through table rows, so they need no alias entry.
* ``tid2eid`` (Hash MoE routing tables) is an edge-keep tensor-level
  special case: it lives under the cloud-owned layers subtree, but
  vllm-ascend reads it directly from the shard files by path, so the edge
  slice must keep it even though the load-time filter would drop it.
* ``hc_head_{fn,base,scale}`` (DeepSeek-V4's hyper-connection head) is a
  both-roles-keep tensor-level special case: the three tensors are FLAT
  parameters on the model (no dot boundary), so no ownership-table row --
  whose matching is all on dot boundaries -- can cover them, and the
  load-time filter never skips them either. Both roles construct the
  parameters (there is no module to prune at build time), so both slices
  keep them; only the cloud actually runs the kernel.
* The source shard ``__metadata__`` (e.g. ``{"format": "pt"}``) is
  preserved in the output shards, keeping the split directories loadable
  by stock HF tooling and not only vLLM.
* Non-weight files follow per-role metadata rules (fnmatch wildcards).
  Directories are NOT copied by default -- only when a rule other than
  ``*`` names one explicitly (e.g. QuaRot's ``optional/``), or with
  ``--metadata all``.
* Tensors matched by NO rule fail fast (exit code 3): classify them
  explicitly via ``--keep-prefix`` / ``--keep-name-regex`` or extend
  ``LWD_MODULE_OWNERSHIP``. Nothing is silently dropped.

Engineering discipline (same as the migrated original):

* Deterministic: all iteration is sorted; two runs over the same source
  produce byte-identical output directories.
* Atomic-ish: shards are written under ``-of-XXXXX`` temp names and renamed
  to ``-of-YYYYY`` only after every write succeeded; the index is written
  last, so a half-written output never looks complete.
* Read-only guard: stat snapshot of ``--src`` before/after; any change
  aborts with exit code 6.
* Index trap: a single-shard output normally drops the index file, but a
  copied companion safetensors (e.g. QuaRot ``rot.safetensors``) makes one
  mandatory -- the index is always rebuilt whenever any tensor file was
  written.

Usage::

    # dry-run first: only print the keep/drop plan, write nothing
    python scripts/lwd/split_lwd_checkpoint.py \\
        --src /hf/Qwen3.5-8B --role edge --dry-run

    # real split (defaults: shards <= 5 GiB, small process pool)
    python scripts/lwd/split_lwd_checkpoint.py \\
        --src /hf/Qwen3.5-8B --dst /hf/Qwen3.5-8B-edge \\
        --role edge --max-shard-gb 8 --jobs 8

Exit codes: 0 success, 1 unexpected error (bug), 2 usage / unusable
source, 3 rule coverage failure, 4 output verification failure, 5 write
failure, 6 source changed during the split.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import multiprocessing as mp
import os
import shutil
import struct
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import regex as re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_NAME = "model.safetensors.index.json"
INDEX_SUFFIX = ".safetensors.index.json"

# Known non-weight safetensors companions: files the runtime reads by path
# instead of through the index (QuaRot rotation matrices, loaded from the
# quant description's rotation map when an EAGLE3 drafter is enabled).
# With an index, any *.safetensors the weight_map does not reference is a
# companion; without one these patterns are the fallback.
COMPANION_SAFETENSORS = (
    "rot.safetensors",
    "*quarot*.safetensors",
    "*rotation*.safetensors",
)

# Never copied as metadata: docs / VCS noise are useless in a serving
# directory, and the output index is rebuilt anyway.
SKIP_FILE_EXACT = ("README.md", "LICENSE", ".gitattributes", "gitattributes")

# Never copied as metadata: name prefixes.
SKIP_FILE_PREFIX = (".git", "__pycache__", INDEX_NAME)

# Non-weight entries per role (fnmatch patterns):
#   * edge is the serving front-end: copy every file, plus the ``optional``
#     directory (QuaRot) which the "*" rule deliberately does not cover;
#   * cloud is a pure decoding engine: only the config (the model graph
#     needs it), the tokenizer family (hard prerequisite for the structured
#     output migration, F3 -- grammar compilation lives on the cloud side
#     and needs the vocab files) and checkpoint companions. Multi-modal
#     processor files are deliberately NOT shipped to the cloud: the cloud
#     never sees raw multi-modal inputs.
# Directories are copied only when a rule other than "*" names them, so
# "optional" is listed explicitly for both roles.
METADATA_RULES: dict[str, tuple[str, ...]] = {
    "edge": ("*", "optional"),
    "cloud": (
        "config.json",
        "configuration.json",
        "*quant*description*.json",
        "chat_template*",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "vocab.*",
        "tokenizer.model",
        "rot.safetensors",
        "optional",
    ),
}

ROLES = ("cloud", "edge")

# I/O tuning: fallback copy chunk (8 MiB) and the per-worker progress
# report granularity (32 MiB).
COPY_CHUNK = 8 * 1024 * 1024
PROGRESS_STEP = 32 * 1024 * 1024
PROGRESS_INTERVAL = 0.2

# ---------------------------------------------------------------------------
# Role rules: ownership table + aliases + tensor-level special cases
# ---------------------------------------------------------------------------

# Vendored snapshot of vllm.model_executor.models.utils.LWD_MODULE_OWNERSHIP
# for hosts without a vllm install (this script is stdlib-only).
# tests/scripts/lwd/test_split_lwd_checkpoint.py asserts the snapshot equals
# the live table -- update both together, never let them drift silently.
_VENDORED_MODULE_OWNERSHIP: dict[str, frozenset[str]] = {
    "model.embed_tokens": frozenset({"edge", "cloud"}),
    "lm_head": frozenset({"edge", "cloud"}),
    "model.layers": frozenset({"cloud"}),
    "model.norm": frozenset({"cloud"}),
    "output_attn_res_norm": frozenset({"cloud"}),
    "output_attn_res_proj": frozenset({"cloud"}),
    "visual": frozenset({"edge"}),
    "vision_tower": frozenset({"edge"}),
    "mm_projector": frozenset({"edge"}),
    "model.mtp": frozenset({"cloud"}),
}

# Tensor-level special cases per role: explicitly registered keeps that
# cannot be expressed as ownership-table rows.
# DeepSeek-V4 Hash MoE routing tables: the edge builds no layers, but
# vllm-ascend reads these tensors straight from the shard files by path
# (vllm_ascend/worker/lwd_hash/lwd_hash_tables.py), so the edge slice must
# keep them even though the load-time filter would drop them.
_EDGE_KEEP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:model\.)?layers\.\d+\.(?:ffn|mlp)\.gate\.tid2eid(?:\.weight)?$"),
)

# Tensor-level special case for BOTH roles: DeepSeek-V4's hyper-connection
# head collapses the hc multi-stream residual at the top of the trunk
# forward, right before the final norm. Its three tensors are FLAT
# parameters on the model (``hc_head_{fn,base,scale}``, no dot boundary),
# so no ownership-table row can express them and the load-time filter never
# skips them. Both roles construct the parameters (there is no module to
# prune at build time), so both slices keep them; only the cloud runs the
# kernel.
_BOTH_KEEP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:model\.)?hc_head_(?:fn|base|scale)(?:\.weight)?$"),
)

# Checkpoint-layout aliases for ownership rows: tensor-name prefixes that
# denote the same module in other naming layouts (native DeepSeek
# checkpoints drop the leading ``model.`` and rename embed/head, cf. the
# weights mapper in vllm/model_executor/models/utils.py). The matcher below
# already covers the canonical layouts (flat ``model.*``, VL
# ``model.language_model.*`` nesting, bare last components); these aliases
# extend coverage so native names do not land in the fail-fast bucket.
_ROW_PREFIX_ALIASES: dict[str, tuple[str, ...]] = {
    "model.embed_tokens": ("embed_tokens.", "embed.", "model.embed."),
    "lm_head": ("lm_head.", "head.", "model.head."),
    "model.layers": ("layers.",),
    "model.norm": ("norm.",),
    # QuaRot-quantized MTP checkpoints export the draft input rotation as a
    # flat top-level ``rot.weight`` (no ``model.`` prefix): the cloud draft
    # remaps it into the spec layer, the edge target discards ``rot.*`` via
    # its weights mapper. It is draft input plumbing, so it rides the MTP
    # row: cloud keeps, edge drops.
    "model.mtp": ("mtp.", "rot."),
}


def load_module_ownership() -> tuple[dict[str, frozenset[str]], bool]:
    """Return the module ownership table.

    Prefers the live ``vllm.model_executor.models.utils.LWD_MODULE_OWNERSHIP``
    (single source of truth, F1) and falls back to the vendored snapshot on
    hosts where vllm is not importable.

    Returns:
        (table, live_import) -- ``live_import`` is True when the table was
        imported from the vllm package.
    """
    try:
        from vllm.model_executor.models.utils import LWD_MODULE_OWNERSHIP
    except ImportError:
        return dict(_VENDORED_MODULE_OWNERSHIP), False
    return {str(k): frozenset(v) for k, v in LWD_MODULE_OWNERSHIP.items()}, True


def _matches_module_suffix(name: str, key: str) -> bool:
    """True when a module prefix of ``name`` equals or ends with ``key``.

    Mirrors the load-time filter (``lwd_weight_filter``): matches the module
    subtree of a pruned tree anywhere in the name, e.g. key ``visual`` owns
    ``model.visual.merger.ln_q.weight``, key ``model.norm`` owns
    ``model.language_model.model.norm.weight`` (after VL unwrap), and key
    ``vision_tower`` owns ``vision_tower.encoder.blocks.0.attn.qkv.weight``
    (Kimi-K3 layout, no unwrap needed).
    """
    parts = name.split(".")
    for i in range(1, len(parts)):
        path = ".".join(parts[:i])
        if path == key or path.endswith("." + key):
            return True
    return False


def _unwrap_vl_text_prefix(name: str) -> str:
    """Strip the VL text-model nesting: ``model.language_model.X`` -> ``model.X``."""
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    return name


def _tensor_matches_module(name: str, key: str) -> bool:
    """Mirror of the load-time filter's per-key match (dot-boundary suffix)."""
    name = _unwrap_vl_text_prefix(name)
    bare = key.rsplit(".", 1)[-1]
    return (
        name == key
        or name.startswith(key + ".")
        or _matches_module_suffix(name, key)
        or name.startswith(bare + ".")
    )


def matched_ownership_rows(name: str, table: Mapping[str, frozenset[str]]) -> set[str]:
    """Ownership rows that own this tensor name (canonical match + aliases)."""
    rows = {key for key in table if _tensor_matches_module(name, key)}
    rows |= {
        row
        for row, aliases in _ROW_PREFIX_ALIASES.items()
        if row in table and any(name.startswith(alias) for alias in aliases)
    }
    return rows


def classify_all(
    weight_map: Mapping[str, str],
    role: str,
    table: Mapping[str, frozenset[str]],
    cli_prefixes: Iterable[str] = (),
    cli_patterns: Iterable[re.Pattern[str]] = (),
) -> tuple[set[str], set[str], list[str], dict[str, list[str]]]:
    """Classify every checkpoint tensor for one role.

    Decision order (first match wins for attribution): CLI overrides ->
    tensor-level special cases -> ownership table. A role keeps a table
    tensor only when it owns EVERY row the name matches, mirroring
    ``lwd_should_skip_weight`` in the load-time filter.

    Returns:
        (keep, drop, uncovered, attribution): ``uncovered`` lists tensors
        matched by no rule at all (the caller must fail fast on them);
        ``attribution`` maps a rule label to its tensor names.
    """
    keep: set[str] = set()
    drop: set[str] = set()
    uncovered: list[str] = []
    attribution: dict[str, list[str]] = {}
    cli_prefixes = tuple(cli_prefixes)
    cli_patterns = tuple(cli_patterns)

    def _attribute(label: str, name: str) -> None:
        attribution.setdefault(label, []).append(name)

    for name in sorted(weight_map):
        matched_prefix = next(
            (prefix for prefix in cli_prefixes if name.startswith(prefix)), None
        )
        if matched_prefix is not None:
            keep.add(name)
            _attribute(f"cli-prefix:{matched_prefix}", name)
            continue
        matched_regex = next((p for p in cli_patterns if p.match(name)), None)
        if matched_regex is not None:
            keep.add(name)
            _attribute(f"cli-regex:{matched_regex.pattern}", name)
            continue
        # Tensor-level special case for BOTH roles: DeepSeek-V4's flat
        # hyper-connection head parameters (no dot boundary -- no ownership
        # row can match them; the load-time filter keeps them for both
        # roles too).
        if any(p.match(_unwrap_vl_text_prefix(name)) for p in _BOTH_KEEP_PATTERNS):
            keep.add(name)
            _attribute("special:hc_head", name)
            continue
        if role == "edge" and any(p.match(name) for p in _EDGE_KEEP_PATTERNS):
            keep.add(name)
            _attribute("special:tid2eid", name)
            continue
        rows = matched_ownership_rows(name, table)
        if not rows:
            uncovered.append(name)
            continue
        label = f"table:{sorted(rows)[0]}"
        if all(role in table[row] for row in rows):
            keep.add(name)
        else:
            drop.add(name)
        _attribute(label, name)
    return keep, drop, uncovered, attribution


# ---------------------------------------------------------------------------
# Error types: expected failures map to process exit codes (see docstring)
# ---------------------------------------------------------------------------


class LwdSplitError(Exception):
    """Expected split failure; ``exit_code`` is the process exit code."""

    exit_code = 1


class SourceError(LwdSplitError):
    exit_code = 2


class RuleCoverageError(LwdSplitError):
    exit_code = 3


class OutputVerifyError(LwdSplitError):
    exit_code = 4


class WriteError(LwdSplitError):
    exit_code = 5


class SourceModifiedError(LwdSplitError):
    exit_code = 6


# ---------------------------------------------------------------------------
# safetensors header parsing (stdlib only, never reads the payload)
# ---------------------------------------------------------------------------


class TensorInfo:
    __slots__ = ("dtype", "shape", "start", "end")

    def __init__(self, dtype: str, shape: list[int], start: int, end: int) -> None:
        self.dtype = dtype
        self.shape = shape
        self.start = start
        self.end = end


def read_shard_header(path: str) -> tuple[dict[str, TensorInfo], dict | None, int]:
    """Parse a safetensors header without reading the payload.

    Layout: ``[u64 LE: header length N][N bytes JSON header][data area]``.

    Returns:
        (tensor name -> TensorInfo, ``__metadata__`` or None, absolute
        offset where the data area starts)
    """
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise SourceError(f"{path}: not a valid safetensors file (too short)")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(header_len))
    tensors: dict[str, TensorInfo] = {}
    metadata = header.get("__metadata__")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        tensors[name] = TensorInfo(info["dtype"], info["shape"], start, end)
    return tensors, metadata, 8 + header_len


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src", required=True, help="complete model directory (with *.safetensors)"
    )
    parser.add_argument(
        "--dst",
        help="output directory (must differ from --src; optional with --dry-run)",
    )
    parser.add_argument(
        "--role", required=True, choices=ROLES, help="which LWD role to slice for"
    )
    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=5.0,
        help="output shard size cap in GiB (default 5; larger = fewer shards)",
    )
    parser.add_argument(
        "--keep-prefix",
        action="append",
        default=[],
        help="extra keep prefix, explicit operator classification (repeatable)",
    )
    parser.add_argument(
        "--keep-name-regex",
        action="append",
        default=[],
        help=(
            "extra keep regex on the tensor name (repeatable; "
            "for cases prefixes cannot express)"
        ),
    )
    parser.add_argument(
        "--metadata",
        choices=("auto", "all"),
        default="auto",
        help=(
            "non-weight entries: auto=per-role rules; "
            "all=copy everything (cloud with unpatched vllm)"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "writer processes; 0=auto (min(4, cpus)), "
            "1=single process (easier debugging)"
        ),
    )
    parser.add_argument(
        "--no-verify-src-readonly",
        action="store_true",
        help=(
            "disable the before/after read-only snapshot check of --src (on by default)"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "disable write progress "
            "(default: in-place line on ttys, throttled otherwise)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print the keep/drop plan, write nothing",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Source inventory
# ---------------------------------------------------------------------------


def find_index_file(src: str) -> str | None:
    """Find the index file (may be named quant_model_weights.safetensors.index.json)."""
    if os.path.isfile(os.path.join(src, INDEX_NAME)):
        return INDEX_NAME
    candidates = sorted(
        name
        for name in os.listdir(src)
        if name.endswith(INDEX_SUFFIX) and os.path.isfile(os.path.join(src, name))
    )
    return candidates[0] if candidates else None


def list_safetensors(src: str) -> list[str]:
    return sorted(name for name in os.listdir(src) if name.endswith(".safetensors"))


def load_weight_map(src: str) -> tuple[dict[str, str], str | None]:
    """Return (tensor name -> shard file name, index file name or None).

    Uses any ``*.safetensors.index.json`` when present; otherwise enumerates
    shard headers one by one (no safetensors package needed).
    """
    index_name = find_index_file(src)
    if index_name is not None:
        with open(os.path.join(src, index_name), encoding="utf-8") as handle:
            return json.load(handle)["weight_map"], index_name
    weight_map: dict[str, str] = {}
    for name in list_safetensors(src):
        # Companions are read by path at runtime, never through the weight
        # map; skipping them here keeps the fallback identical to the
        # indexed path (whose weight_map also excludes companion tensors).
        if any(fnmatch.fnmatch(name, pat) for pat in COMPANION_SAFETENSORS):
            continue
        tensors, _, _ = read_shard_header(os.path.join(src, name))
        for key in tensors:
            weight_map[key] = name
    return weight_map, None


def classify_safetensors(
    src: str, weight_map: dict[str, str], index_name: str | None
) -> tuple[set[str], list[str]]:
    """Split ``*.safetensors`` into weight shards and companion files.

    Returns:
        (weight shard file names, companion safetensors file names)
    """
    files = list_safetensors(src)
    if index_name is not None:
        weights = set(weight_map.values())
        return weights, [name for name in files if name not in weights]
    # No index: fall back to the known companion patterns so rotation
    # matrices are never treated as weight shards.
    companions = [
        name
        for name in files
        if any(fnmatch.fnmatch(name, pat) for pat in COMPANION_SAFETENSORS)
    ]
    weights = [name for name in files if name not in companions]
    return set(weights), companions


# ---------------------------------------------------------------------------
# Shard planning
# ---------------------------------------------------------------------------


def collect_tensor_info(
    src: str, keep: set[str], weight_files: set[str]
) -> tuple[dict[str, TensorInfo], dict | None, int]:
    """Read headers of source shards and collect info for kept tensors.

    Returns:
        (name -> TensorInfo, first shard's ``__metadata__`` or None, payload
        bytes the output must write)
    """
    info: dict[str, TensorInfo] = {}
    metadata: dict | None = None
    total = 0
    for shard in sorted(weight_files):
        tensors, shard_metadata, base = read_shard_header(os.path.join(src, shard))
        if shard_metadata is not None and metadata is None:
            metadata = shard_metadata
        for name, tensor in tensors.items():
            if name in keep:
                if name in info:
                    raise SourceError(f"tensor {name!r} appears in multiple shards")
                # TensorInfo from the parser holds offsets relative to the
                # data area; make them absolute for the copy step.
                info[name] = TensorInfo(
                    tensor.dtype, tensor.shape, base + tensor.start, base + tensor.end
                )
                total += tensor.end - tensor.start
    missing = sorted(keep - set(info))
    if missing:
        suffix = "..." if len(missing) > 5 else ""
        raise SourceError(f"tensors missing from shards: {missing[:5]}{suffix}")
    return info, metadata, total


def plan_shards(
    info: Mapping[str, TensorInfo], max_bytes: int
) -> list[tuple[str, list[str]]]:
    """Pack kept tensors into output shards of at most ``max_bytes``.

    Deterministic: input is iterated in sorted order, so two runs over the
    same source produce identical shard contents. Oversized single tensors
    land in a shard of their own.

    Returns:
        list of (temp shard name, tensor names)
    """
    wanted: dict[str, list[str]] = {}
    cur_name = "model-00001-of-XXXXX.safetensors"
    cur_size = 0
    count = 1
    for name in sorted(info):
        size = info[name].end - info[name].start
        if cur_size > 0 and cur_size + size > max_bytes:
            count += 1
            cur_name = f"model-{count:05d}-of-XXXXX.safetensors"
            cur_size = 0
        wanted.setdefault(cur_name, []).append(name)
        cur_size += size
    return [(name, wanted[name]) for name in sorted(wanted)]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_all(fd: int, data: memoryview, offset: int) -> None:
    view = data
    while view:
        written = os.pwrite(fd, view, offset)
        if not written:
            raise WriteError(f"short write at offset {offset}")
        offset += written
        view = view[written:]


def _copy_range(src_fd: int, dst_fd: int, start: int, end: int, offset: int) -> None:
    """Copy ``[start, end)`` of ``src_fd`` to ``offset`` in ``dst_fd``.

    Tries zero-copy ``copy_file_range`` first (requires file descriptors),
    falls back to chunked ``pread`` when the kernel refuses (e.g. older
    kernels, cross-filesystem).
    """
    copied = 0
    while copied < end - start:
        try:
            n = os.copy_file_range(
                src_fd,
                dst_fd,
                count=min(COPY_CHUNK, end - start - copied),
                offset_src=start + copied,
                offset_dst=offset + copied,
            )
            if not n:
                raise WriteError(f"fd {src_fd}: unexpected EOF at {start + copied}")
        except OSError:
            remaining = min(COPY_CHUNK, end - start - copied)
            chunk = os.pread(src_fd, remaining, start + copied)
            if not chunk:
                raise WriteError(
                    f"fd {src_fd}: unexpected EOF at {start + copied}"
                ) from None
            _write_all(dst_fd, memoryview(chunk), offset + copied)
            n = len(chunk)
        copied += n


def write_one_shard(
    shard: tuple[str, list[str]],
    src: str,
    dst: str,
    info: Mapping[str, TensorInfo],
    weight_map: Mapping[str, str],
    queue: object | None = None,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write one output shard: JSON header + payload copied from source shards.

    The source ``__metadata__`` (e.g. ``{"format": "pt"}``) is preserved so
    the output stays loadable by stock HF tooling. With ``queue`` (a
    multiprocessing manager queue) the bytes written are pushed at least
    every ``PROGRESS_STEP`` bytes for progress reporting.
    """
    name, wanted = shard
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for tensor in wanted:
        size = info[tensor].end - info[tensor].start
        header[tensor] = {
            "dtype": info[tensor].dtype,
            "shape": info[tensor].shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if metadata:
        header["__metadata__"] = dict(metadata)
    body = json.dumps(header, separators=(",", ":")).encode()
    body += b" " * ((-len(body)) % 8)
    raw = struct.pack("<Q", len(body)) + body
    with open(os.path.join(dst, name), "wb") as handle:
        _write_all(handle.fileno(), memoryview(raw), 0)
        pos = len(raw)
        pending = 0
        src_fds: dict[str, int] = {}
        try:
            for tensor in wanted:
                tensor_info = info[tensor]
                source_shard = weight_map[tensor]
                if source_shard not in src_fds:
                    src_fds[source_shard] = os.open(
                        os.path.join(src, source_shard), os.O_RDONLY
                    )
                _copy_range(
                    src_fds[source_shard],
                    handle.fileno(),
                    tensor_info.start,
                    tensor_info.end,
                    pos,
                )
                size = tensor_info.end - tensor_info.start
                pos += size
                pending += size
                if queue is not None and pending >= PROGRESS_STEP:
                    queue.put(pending)
                    pending = 0
            os.fsync(handle.fileno())
        finally:
            for fd in src_fds.values():
                os.close(fd)
    if queue is not None and pending:
        queue.put(pending)


def write_slices(
    src: str,
    dst: str,
    entries: list[tuple[str, list[str]]],
    info: Mapping[str, TensorInfo],
    weight_map: Mapping[str, str],
    jobs: int,
    progress: Progress,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write every planned shard (optionally with a process pool).

    Temp names end in ``-of-XXXXX``; once all writes succeeded the files are
    atomically renamed to ``-of-YYYYY`` with the true shard count, so a
    half-written output is never mistaken for a complete one. On failure the
    temp files are removed -- the output directory keeps only the metadata
    copies and can simply be re-run.
    """
    if jobs < 0:
        raise SourceError(f"--jobs must be >= 0, got {jobs}")

    def _drain(queue: object, progress: Progress) -> None:
        while True:
            try:
                progress.add(queue.get_nowait())
            except Exception:
                break

    def _write_serial() -> None:
        for shard in entries:
            write_one_shard(shard, src, dst, info, weight_map, metadata=metadata)
            progress.add(sum(info[t].end - info[t].start for t in shard[1]))
        progress.finish()

    try:
        if jobs in (0, 1) or len(entries) <= 1:
            _write_serial()
        else:
            try:
                with mp.Manager() as manager:
                    queue = manager.Queue()
                    with ProcessPoolExecutor(max_workers=jobs) as pool:
                        futures = [
                            pool.submit(
                                write_one_shard,
                                shard,
                                src,
                                dst,
                                info,
                                weight_map,
                                queue,
                                metadata,
                            )
                            for shard in entries
                        ]
                        pending = set(futures)
                        while pending:
                            done, pending = wait(
                                pending, timeout=0.3, return_when=FIRST_COMPLETED
                            )
                            for future in done:
                                future.result()
                            _drain(queue, progress)
                    progress.finish()
            except (OSError, RuntimeError, ImportError) as exc:
                print(
                    f"[warn] parallel writer unavailable ({exc!r}); "
                    "falling back to single process",
                    file=sys.stderr,
                )
                _write_serial()
    except BaseException:
        for name, _ in entries:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(dst, name))
        raise

    # Atomic rename to the true shard total once every write succeeded.
    total = len(entries)
    for name, _ in entries:
        final = name.replace("-of-XXXXX", f"-of-{total:05d}")
        os.replace(os.path.join(dst, name), os.path.join(dst, final))


def write_index(
    dst: str,
    entries: list[tuple[str, list[str]]],
    companion_present: bool,
    total_bytes: int,
) -> None:
    """Write (or deliberately drop) the output index.

    The index is rebuilt whenever any tensor file was written. A
    single-shard output normally needs no index, but a copied companion
    safetensors makes one mandatory -- without it the loader would treat
    the companion as the only shard and fail on every tensor.
    """
    total = len(entries)
    if total <= 1 and not companion_present:
        index_path = os.path.join(dst, INDEX_NAME)
        if os.path.exists(index_path):
            os.remove(index_path)
        print(
            "[index] single shard, no companions: no index needed (removed if present)"
        )
        return
    weight_map_out: dict[str, str] = {}
    for name, tensors in entries:
        final = name.replace("-of-XXXXX", f"-of-{total:05d}")
        for tensor in tensors:
            weight_map_out[tensor] = final
    payload = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map_out}
    index_path = os.path.join(dst, INDEX_NAME)
    tmp = index_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, index_path)
    print(
        f"[index] wrote {INDEX_NAME}: {total} shard(s), {len(weight_map_out)} tensors"
    )


def verify_output(
    dst: str, entries: list[tuple[str, list[str]]], info: Mapping[str, TensorInfo]
) -> None:
    """Re-read the written headers and confirm tensor sets and dtype/shape.

    Also checks each shard's file size against its own header (8 + header
    length + payload), a cheap guard against truncated or shifted writes.
    """
    total = len(entries)
    expected: set[str] = set()
    for name, tensors in entries:
        final = name.replace("-of-XXXXX", f"-of-{total:05d}")
        path = os.path.join(dst, final)
        if not os.path.isfile(path):
            raise OutputVerifyError(f"missing output shard: {final}")
        on_disk, _, base = read_shard_header(path)
        if set(on_disk) != set(tensors):
            raise OutputVerifyError(
                f"{final}: tensor set mismatch "
                f"(expected {len(tensors)}, got {len(on_disk)})"
            )
        payload = sum(on_disk[t].end - on_disk[t].start for t in tensors)
        size = os.path.getsize(path)
        if size != base + payload:
            raise OutputVerifyError(
                f"{final}: file size {size} != header+payload {base + payload}"
            )
        for tensor in tensors:
            if on_disk[tensor].dtype != info[tensor].dtype or list(
                on_disk[tensor].shape
            ) != list(info[tensor].shape):
                raise OutputVerifyError(f"{final}: dtype/shape mismatch for {tensor}")
        expected |= set(tensors)
    if expected != set(info):
        raise OutputVerifyError("union of shard tensors != planned tensors")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _metadata_candidates(
    src: str, weight_files: set[str], companions: list[str]
) -> tuple[list[str], list[str]]:
    """Return (metadata file candidates, directory candidates).

    Files start with the companions (so they can never fall under the
    skip rules), then every top-level entry that is neither a weight shard
    nor a non-companion ``*.safetensors`` nor an index (any name, standard
    or custom -- the output always rebuilds its own) nor a skipped name.
    """
    files = list(companions)
    dirs: list[str] = []
    for name in sorted(os.listdir(src)):
        if name in weight_files or name in files:
            continue
        if (
            name.endswith(".safetensors")
            or name.endswith(INDEX_SUFFIX)
            or name.startswith(SKIP_FILE_PREFIX)
        ):
            continue
        if name in SKIP_FILE_EXACT:
            continue
        if os.path.isdir(os.path.join(src, name)):
            dirs.append(name)
        else:
            files.append(name)
    return sorted(files), dirs


def plan_metadata(
    src: str, role: str, mode: str, weight_files: set[str], companions: list[str]
) -> tuple[list[str], list[str]]:
    """Return (entries to copy, entries deliberately skipped) for this role.

    Files follow the per-role fnmatch rules. Directories are NOT copied by
    default -- only when a rule other than ``*`` names one explicitly
    (e.g. QuaRot's ``optional/``), or with ``mode == "all"``.
    """
    allowed = ("*",) if mode == "all" else METADATA_RULES.get(role, ())
    files, dirs = _metadata_candidates(src, weight_files, companions)
    wanted = [name for name in files if any(fnmatch.fnmatch(name, p) for p in allowed)]
    wanted += [
        name
        for name in dirs
        if mode == "all" or any(p != "*" and fnmatch.fnmatch(name, p) for p in allowed)
    ]
    skipped = [name for name in files + dirs if name not in wanted]
    return wanted, skipped


def copy_metadata(src: str, dst: str, names: list[str]) -> list[str]:
    """Copy non-weight entries; directories are copied recursively."""
    copied: list[str] = []
    for name in names:
        source = os.path.join(src, name)
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(dst, name), dirs_exist_ok=True)
            copied.append(name + "/")
        else:
            shutil.copy2(source, os.path.join(dst, name))
            copied.append(name)
    return copied


# ---------------------------------------------------------------------------
# Read-only guard
# ---------------------------------------------------------------------------


def snapshot_tree(path: str) -> dict[str, tuple[int, int, int]]:
    """Stat snapshot of every file under ``path``: name -> (mtime_ns, size, inode)."""
    snap: dict[str, tuple[int, int, int]] = {}
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(dirs)
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            try:
                st = os.stat(full)
            except OSError:
                continue
            snap[rel] = (int(st.st_mtime_ns), int(st.st_size), int(st.st_ino))
    return snap


def snapshot_changes(
    before: dict[str, tuple[int, int, int]], after: dict[str, tuple[int, int, int]]
) -> list[str]:
    """Paths whose snapshot entry changed, was added or removed (sorted)."""
    return [
        name
        for name in sorted(set(before) | set(after))
        if before.get(name) != after.get(name)
    ]


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


class Progress:
    """Stderr-only progress line: in-place on a tty, throttled otherwise."""

    def __init__(self, total: float, enabled: bool = True) -> None:
        self.total = max(float(total), 1.0)
        self.done = 0
        self.enabled = enabled and sys.stderr.isatty()
        self.start = time.time()
        self.last = 0.0

    def add(self, n: int) -> None:
        self.done += n
        if not self.enabled:
            return
        now = time.time()
        if now - self.last < PROGRESS_INTERVAL:
            return
        self.last = now
        pct = min(100.0, self.done * 100.0 / self.total)
        sys.stderr.write(
            f"\r[write] {pct:5.1f}% ({fmt_gib(self.done)}/{fmt_gib(int(self.total))})"
        )
        sys.stderr.flush()

    def finish(self) -> None:
        if not self.enabled:
            return
        elapsed = time.time() - self.start
        sys.stderr.write(
            f"\r[write] 100.0% ({fmt_gib(int(self.total))}) in {elapsed:.1f}s   \n"
        )
        sys.stderr.flush()


def fmt_gib(n: int) -> str:
    return f"{n / (1 << 30):.2f} GiB"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _parse_cli_patterns(raw: list[str]) -> list[re.Pattern[str]]:
    out = []
    for item in raw:
        try:
            out.append(re.compile(item))
        except re.error as exc:
            raise SourceError(f"--keep-name-regex {item!r}: {exc}") from exc
    return out


def _print_rule_report(
    role: str,
    table: Mapping[str, frozenset[str]],
    live: bool,
    keep: set[str],
    drop: set[str],
    uncovered: list[str],
    attribution: dict[str, list[str]],
) -> None:
    table_source = (
        "live import from vllm" if live else "vendored fallback (vllm not importable)"
    )
    keep_rows = sorted(k for k, owners in table.items() if role in owners)
    drop_rows = sorted(k for k, owners in table.items() if role not in owners)
    print(f"[rule] module ownership table: {table_source} ({len(table)} rows)")
    print(f"[rule] {role} keeps rows: {', '.join(keep_rows)}")
    print(f"[rule] {role} drops rows: {', '.join(drop_rows)}")
    if role == "edge" and _EDGE_KEEP_PATTERNS:
        print(
            "[rule] edge tensor special-case keeps: tid2eid (Hash MoE routing tables)"
        )
    if _BOTH_KEEP_PATTERNS:
        print(
            "[rule] both-role tensor special-case keeps: "
            "hc_head (DeepSeek-V4 hyper-connection head)"
        )
    total = len(keep) + len(drop) + len(uncovered)
    print(
        f"[plan] {total} tensors: keep {len(keep)}, drop {len(drop)}, "
        f"uncovered {len(uncovered)}"
    )
    for label in sorted(attribution):
        print(f"[plan] {label}: {len(attribution[label])} tensors")
    if drop:
        groups: dict[str, int] = {}
        for name in drop:
            key = ".".join(name.split(".")[:3])
            groups[key] = groups.get(key, 0) + 1
        top = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        print("[drop] top dropped groups (first 3 components):")
        for key, count in top:
            print(f"    {key}: {count}")
    if uncovered:
        print(f"[rule] FAIL: {len(uncovered)} tensor(s) matched by no rule:")
        for name in uncovered[:20]:
            print(f"    {name}")
        groups = {}
        for name in uncovered:
            key = ".".join(name.split(".")[:3])
            groups[key] = groups.get(key, 0) + 1
        top = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        print("[rule] uncovered groups (first 3 components):")
        for key, count in top:
            print(f"    {key}: {count}")
        print(
            "[rule] classify them explicitly with --keep-prefix/--keep-name-regex "
            "or extend LWD_MODULE_OWNERSHIP"
        )


def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cli_patterns = _parse_cli_patterns(args.keep_name_regex)

    table, live = load_module_ownership()
    if not live:
        print(
            "[warn] vllm not importable: using the vendored ownership table snapshot",
            file=sys.stderr,
        )

    src = os.path.realpath(os.path.expanduser(args.src))
    if not os.path.isdir(src):
        raise SourceError(f"--src {args.src!r}: not a directory")
    guard_before = None if args.no_verify_src_readonly else snapshot_tree(src)

    weight_map, index_name = load_weight_map(src)
    if not weight_map:
        raise SourceError(f"--src {src!r}: no tensors found")
    weight_files, companions = classify_safetensors(src, weight_map, index_name)

    keep, drop, uncovered, attribution = classify_all(
        weight_map, args.role, table, args.keep_prefix, cli_patterns
    )
    _print_rule_report(args.role, table, live, keep, drop, uncovered, attribution)
    if uncovered:
        raise RuleCoverageError(
            f"{len(uncovered)} tensor(s) matched no rule "
            f"(first: {uncovered[0]!r}); classify them explicitly before splitting"
        )
    if not keep:
        raise RuleCoverageError("no tensors matched the keep rules for this role")

    info, shard_metadata, payload_bytes = collect_tensor_info(src, keep, weight_files)
    entries = plan_shards(info, int(args.max_shard_gb * (1 << 30)))
    print(
        f"[plan] {args.role} output: {len(entries)} shard(s), "
        f"{fmt_gib(payload_bytes)} payload, shard cap {args.max_shard_gb} GiB"
    )

    meta_files, meta_skipped = plan_metadata(
        src, args.role, args.metadata, weight_files, companions
    )
    print(f"[meta] copying {len(meta_files)} entries for role {args.role}:")
    for name in meta_files:
        print(f"    {name}")
    if meta_skipped:
        print(f"[meta] skipping {len(meta_skipped)}: {', '.join(meta_skipped)}")
    if args.metadata == "auto" and args.role == "cloud":
        print(
            "[meta] cloud minimal set assumes the LWD vllm patches; "
            "for an unpatched vllm use --metadata all"
        )

    if args.dry_run:
        print("[done ] dry-run: nothing written")
        return 0

    if not args.dst:
        raise SourceError("--dst is required unless --dry-run is set")
    dst = os.path.realpath(os.path.expanduser(args.dst))
    if dst == src:
        raise SourceError("--dst must differ from --src")
    if os.path.commonpath([dst, src]) == src:
        raise SourceError("--dst must not be inside --src")
    os.makedirs(dst, exist_ok=True)

    # Disk space preflight: payload + headers + metadata + fixed slack.
    needed = payload_bytes + 256 * 2**20
    free = shutil.disk_usage(dst).free
    if free < needed:
        raise WriteError(
            f"insufficient disk space: need {fmt_gib(needed)}, free {fmt_gib(free)}"
        )

    print(f"[copy] metadata -> {dst}")
    copied = copy_metadata(src, dst, meta_files)
    print(f"[copy] {len(copied)} metadata entries copied: {', '.join(copied)}")

    progress = Progress(payload_bytes, enabled=not args.no_progress)
    jobs = args.jobs if args.jobs > 0 else min(4, os.cpu_count() or 1)
    try:
        write_slices(
            src, dst, entries, info, weight_map, jobs, progress, shard_metadata
        )
        companion_present = any(name in meta_files for name in companions)
        write_index(dst, entries, companion_present, payload_bytes)
    except OSError as exc:
        raise WriteError(f"write failed: {exc}") from exc
    verify_output(dst, entries, info)

    if guard_before is not None:
        changes = snapshot_changes(guard_before, snapshot_tree(src))
        if changes:
            shown = ", ".join(changes[:5])
            raise SourceModifiedError(f"--src changed during the split: {shown}")

    total_out = sum(
        os.path.getsize(os.path.join(dst, f))
        for f in os.listdir(dst)
        if os.path.isfile(os.path.join(dst, f))
    )
    print(f"[done ] role={args.role} -> {dst} ({fmt_gib(total_out)})")
    print(f"[next ] start vllm with --model {dst}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except LwdSplitError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
