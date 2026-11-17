# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LWD role-based module skipping at model construction time.

Design doc: docs/design/lwd_role_module_skip.md (F1, plan A). The
``LWD_MODULE_OWNERSHIP`` table and the ``lwd_skips_module`` pure function
drive per-role construction guards: the edge skips the final norm (the
cloud norm hidden states before sending them DOWN), the cloud skips the
vision tower (multimodal processing lives on the edge).
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import DeepseekV2Config
from transformers import Qwen3Config as Qwen3TextConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
)

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import (
    LWD_MODULE_OWNERSHIP,
    PPMissingLayer,
    lwd_role,
    lwd_role_of,
    lwd_skips_module,
)
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig


def _lwd_config(role: str | None) -> SimpleNamespace | None:
    return SimpleNamespace(role=role) if role is not None else None


def _make_model_config(config: Any) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        dtype=torch.float32,
        max_model_len=128,
        is_diffusion=False,
        head_dtype=None,
        is_mm_prefix_lm=False,
        multimodal_config=None,
        quantization_config=None,
        runner_type="generate",
        is_moe=True,
        logits_processors=None,
        rswa_window=None,
    )


def _build_model(model_cls: type, config: Any, role: str | None) -> torch.nn.Module:
    vllm_config = VllmConfig()
    vllm_config.model_config = _make_model_config(config)
    vllm_config.lwd_config = _lwd_config(role)
    with set_current_vllm_config(vllm_config):
        return model_cls(vllm_config=vllm_config)


def _param_names(model: torch.nn.Module) -> set[str]:
    return {name for name, _ in model.named_parameters()}


@pytest.mark.parametrize(
    ("module_path", "role", "expected_skip"),
    [
        # Dot-boundary suffix match across layouts (CausalLM and VL prefixes).
        ("model.norm", "edge", True),
        ("language_model.model.norm", "edge", True),
        ("model.norm", "cloud", False),
        # Tower-internal norms die with the visual tree, not with model.norm.
        ("visual.merger.norm", "edge", False),
        # Bare attribute names must not match the norm rule.
        ("norm", "edge", False),
        # The whole visual tree belongs to the edge. The tree dies via the
        # root cut: tower-internal module paths are never queried, they are
        # simply never built.
        ("visual", "cloud", True),
        ("model.visual", "cloud", True),
        ("language_model.visual", "cloud", True),
        ("visual.merger.proj", "cloud", False),
        ("visual", "edge", False),
        # Kimi-K3/K2.5 name the tower and projector without the "visual"
        # prefix; the dedicated rows cover the flat and the prefixed layouts.
        ("vision_tower", "cloud", True),
        ("vision_tower", "edge", False),
        ("model.vision_tower", "cloud", True),
        ("mm_projector", "cloud", True),
        ("language_model.model.mm_projector", "cloud", True),
        ("mm_projector", "edge", False),
        # Kimi-K3's final-norm attention-residual tail belongs to the cloud.
        ("output_attn_res_norm", "edge", True),
        ("output_attn_res_norm", "cloud", False),
        ("language_model.model.output_attn_res_proj", "edge", True),
        ("language_model.model.output_attn_res_proj", "cloud", False),
        # Both-role modules.
        ("model.embed_tokens", "edge", False),
        ("model.embed_tokens", "cloud", False),
        ("lm_head", "edge", False),
        ("lm_head", "cloud", False),
        # Transformer layers live on the cloud (edge builds zero layers).
        ("model.layers", "edge", True),
        ("model.layers", "cloud", False),
        # The MTP drafter lives on the cloud.
        ("model.mtp", "edge", True),
        ("model.mtp", "cloud", False),
    ],
)
def test_lwd_skips_module_ownership(
    module_path: str, role: str, expected_skip: bool
) -> None:
    assert lwd_skips_module(module_path, role) is expected_skip


def test_lwd_skips_module_none_role_keeps_pre_lwd_behavior() -> None:
    """Non-LWD processes (role=None) must never skip any module."""
    for module_path in (
        "model.norm",
        "visual",
        "vision_tower",
        "mm_projector",
        "output_attn_res_norm",
        "model.layers",
        "model.mtp",
    ):
        assert lwd_skips_module(module_path, None) is False


def test_lwd_module_ownership_entries_are_explicit() -> None:
    for suffix, owners in LWD_MODULE_OWNERSHIP.items():
        assert owners, f"{suffix} has no owning role"
        assert owners <= {"edge", "cloud"}, f"{suffix} has unknown roles"


def test_lwd_role_is_none_without_lwd_config() -> None:
    with set_current_vllm_config(VllmConfig()):
        assert lwd_role() is None


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_lwd_role_reads_current_config(role: str) -> None:
    vllm_config = VllmConfig()
    vllm_config.lwd_config = _lwd_config(role)
    with set_current_vllm_config(vllm_config):
        assert lwd_role() == role


def test_lwd_role_of_reads_persistent_config_after_init_window() -> None:
    # Regression for the profile_run guard: worker RPCs run after the
    # model-init `set_current_vllm_config` context exits, so the
    # process-global lwd_role() reads None there. The guard must read the
    # role off the runner's own persistent vllm_config instead.
    vllm_config = VllmConfig()
    vllm_config.lwd_config = _lwd_config("cloud")

    assert lwd_role() is None  # the model-init context has exited
    assert lwd_role_of(None) is None
    assert lwd_role_of(VllmConfig()) is None
    assert lwd_role_of(vllm_config) == "cloud"
    assert lwd_skips_module("visual", lwd_role_of(vllm_config))


def test_lwd_cloud_budget_is_empty_without_building_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: cloud startup crashed building the multimodal budget.

    The cloud weight split omits the HF processor files, so building the
    processor there fails (OSError: Can't load image processor ...). The
    budget must stay empty instead of touching the processor.
    """
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.multimodal.registry import MultiModalRegistry

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the cloud role must not build the HF processor")

    monkeypatch.setattr(MultiModalRegistry, "create_processor", _fail)

    vllm_config = VllmConfig()
    vllm_config.lwd_config = _lwd_config("cloud")
    budget = MultiModalBudget(vllm_config)

    assert budget.cache is None
    assert budget.encoder_compute_budget == 0
    assert budget.encoder_cache_size == 0
    assert dict(budget.mm_limits) == {}
    assert dict(budget.mm_max_toks_per_item) == {}
    assert dict(budget.mm_max_items_per_prompt) == {}
    assert dict(budget.mm_max_items_per_batch) == {}
    budget.reset_cache()  # no-op with an empty budget; must not raise


def _build_input_processor(
    monkeypatch: pytest.MonkeyPatch,
    role: str | None,
    budget_factory: Any,
) -> Any:
    """Build an InputProcessor with the renderer and registry stubbed out."""
    import vllm.v1.engine.input_processor as ip_mod

    monkeypatch.setattr(ip_mod, "MultiModalBudget", budget_factory)

    vllm_config = VllmConfig()
    vllm_config.lwd_config = _lwd_config(role)
    vllm_config.model_config = SimpleNamespace(
        try_get_generation_config=lambda: {},
    )
    renderer = SimpleNamespace(_executor=None)
    mm_registry = SimpleNamespace(supports_multimodal_inputs=lambda cfg: True)
    return ip_mod.InputProcessor(vllm_config, renderer, mm_registry=mm_registry)


def test_lwd_cloud_input_processor_skips_the_mm_budget_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the cloud APIServer frontend crashed probing the budget.

    InputProcessor used to build MultiModalBudget unconditionally for mm
    models, which loads the HF processor that the cloud split omits.
    """

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the cloud role must not probe the mm budget")

    processor = _build_input_processor(monkeypatch, "cloud", _fail)

    assert processor.supports_mm_inputs
    assert processor.mm_encoder_cache_size == 0
    assert processor.skip_prompt_length_check is False


@pytest.mark.parametrize("role", ["edge", None])
def test_lwd_edge_input_processor_probes_the_mm_budget(
    monkeypatch: pytest.MonkeyPatch, role: str | None
) -> None:
    """Edge and non-LWD deployments own the vision tower and must keep
    deriving the budget values from the real processor."""

    class _FakeBudget:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.encoder_cache_size = 7
            self.processor = SimpleNamespace(
                info=SimpleNamespace(skip_prompt_length_check=True)
            )

        def reset_cache(self) -> None:
            pass

    processor = _build_input_processor(monkeypatch, role, _FakeBudget)

    assert processor.mm_encoder_cache_size == 7
    assert processor.skip_prompt_length_check is True


def _make_deepseek_v2_config() -> DeepseekV2Config:
    return DeepseekV2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        first_k_dense_replace=0,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=32,
        q_lora_rank=None,
        kv_lora_rank=0,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
    )


@pytest.mark.parametrize("role", ["edge", "cloud", None])
def test_deepseek_v2_norm_guard(role: str | None, dist_init: None) -> None:
    """Edge builds no final norm; cloud and non-LWD keep it."""
    model = _build_model(DeepseekV2Model, _make_deepseek_v2_config(), role)
    names = _param_names(model)
    norm_params = [n for n in names if ".norm." in n]
    layer_params = [n for n in names if ".layers." in n]
    embed_params = [n for n in names if ".embed_tokens." in n]
    if role == "edge":
        assert isinstance(model.norm, PPMissingLayer)
        assert not norm_params
        assert not layer_params
    else:
        assert not isinstance(model.norm, PPMissingLayer)
        assert norm_params
        assert layer_params
    assert embed_params


def _make_qwen3_5_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["full_attention"],
    )


@pytest.mark.parametrize("role", ["edge", "cloud", None])
def test_qwen3_5_norm_guard(role: str | None, dist_init: None) -> None:
    """Edge builds no final norm; cloud and non-LWD keep it."""
    model = _build_model(Qwen3_5Model, _make_qwen3_5_config(), role)
    names = _param_names(model)
    norm_params = [n for n in names if ".norm." in n]
    layer_params = [n for n in names if ".layers." in n]
    embed_params = [n for n in names if ".embed_tokens." in n]
    if role == "edge":
        assert isinstance(model.norm, PPMissingLayer)
        assert not norm_params
        assert not layer_params
    else:
        assert not isinstance(model.norm, PPMissingLayer)
        assert norm_params
        assert layer_params
    assert embed_params


class _DummyVisionTower(torch.nn.Module):
    """Construction seam: stands in for the heavy vision tower internals."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)


class _DummyLanguageModel(torch.nn.Module):
    """Construction seam: stands in for the text stack built inside VL."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(16, 16)
        self.make_empty_intermediate_tensors = lambda *args, **kwargs: None


def _make_qwen3_vl_config() -> Qwen3VLConfig:
    vision_config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=1,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        out_hidden_size=16,
        deepstack_visual_indexes=[0],
    )
    text_config = Qwen3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
    )
    return Qwen3VLConfig(vision_config=vision_config, text_config=text_config)


def _build_qwen3_vl(role: str | None) -> Qwen3VLForConditionalGeneration:
    mm_config = SimpleNamespace(
        mm_encoder_tp_mode="tensor",
        mm_encoder_only=False,
        get_limit_per_prompt=lambda modality: 1,
        get_video_pruning_spec=lambda: None,
        is_multimodal_pruning_enabled=lambda: False,
        video_pruning_rate=0.0,
    )
    config = _make_qwen3_vl_config()
    model_config = SimpleNamespace(
        hf_config=config,
        hf_text_config=config.text_config,
        dtype=torch.float32,
        max_model_len=128,
        is_diffusion=False,
        head_dtype=None,
        is_mm_prefix_lm=False,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
        skip_tokenizer_init=True,
        quantization_config=None,
        runner_type="generate",
        is_moe=False,
        logits_processors=None,
        rswa_window=None,
        is_multimodal_model=True,
        get_model_arch_config=lambda: None,
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = model_config
    vllm_config.lwd_config = _lwd_config(role)
    with set_current_vllm_config(vllm_config):
        return Qwen3VLForConditionalGeneration(vllm_config=vllm_config)


@pytest.mark.parametrize("role", ["edge", "cloud", None])
def test_qwen3_vl_visual_guard(
    role: str | None, dist_init: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cloud builds no vision tower (and no deepstack); edge and non-LWD keep it."""
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3_VisionTransformer",
        _DummyVisionTower,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3LLMForCausalLM",
        _DummyLanguageModel,
    )
    model = _build_qwen3_vl(role)
    names = _param_names(model)
    visual_params = [n for n in names if ".visual." in n]
    if role == "cloud":
        assert isinstance(model.visual, PPMissingLayer)
        assert not visual_params
        # Deepstack lives inside the tower and dies with it.
        assert model.use_deepstack is False
    else:
        assert isinstance(model.visual, _DummyVisionTower)
        assert visual_params
        assert model.use_deepstack is True


def test_kimi_k3_exposes_down_norm_hook() -> None:
    """Both K3 entry architectures implement the optional DOWN-norm hook.

    The LWD cloud runner looks the hook up by name and falls back to identity,
    so a missing implementation would silently ship pre-norm hidden states
    DOWN while the edge skips its norm -> un-normalized logits.
    """
    from vllm.models.kimi_k3.nvidia.model import (
        KimiK3ForConditionalGeneration,
        KimiLinearForCausalLM,
    )

    for cls in (KimiK3ForConditionalGeneration, KimiLinearForCausalLM):
        assert callable(getattr(cls, "finalize_down_hidden", None)), cls.__name__


def test_kimi_k3_down_norm_hook_applies_final_norm() -> None:
    """The hook must apply the model's final norm (not a copy of it)."""
    from vllm.models.kimi_k3.nvidia.model import KimiLinearForCausalLM

    calls: list[tuple[torch.Tensor, object]] = []

    def _norm(hidden: torch.Tensor, residual: object) -> torch.Tensor:
        calls.append((hidden, residual))
        return hidden * 2.0

    fake_self = SimpleNamespace(model=SimpleNamespace(norm=_norm))
    hidden = torch.ones(1)
    out = KimiLinearForCausalLM.finalize_down_hidden(fake_self, hidden)
    assert out.item() == 2.0
    assert calls and calls[0][0] is hidden and calls[0][1] is None


# ---------------------------------------------------------------------------
# Kimi-K3 construction-guard wiring (source scan).
#
# Building the real K3 model needs the full multimodal stack, so the wiring
# contract is locked with an AST scan instead (the same approach as
# vllm-ascend's tests/ut/models/test_lwd_kimi_k3_guard.py): every guard label
# must be a literal that exists as an ``LWD_MODULE_OWNERSHIP`` row, and each
# implementation must guard exactly the modules it owns.
# ---------------------------------------------------------------------------

_K3_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Guard labels each vllm implementation must pass to ``lwd_skips_module``.
#: ``output_attn_res_proj`` shares the norm's guard branch (one call, two
#: assignments), so it has no call site of its own.
_K3_GUARD_LABELS_BY_FILE: dict[str, frozenset[str]] = {
    "vllm/models/kimi_k3/nvidia/model.py": frozenset(
        {"model.norm", "output_attn_res_norm", "vision_tower", "mm_projector"}
    ),
    "vllm/models/kimi_k3/amd/linear.py": frozenset(
        {"model.norm", "output_attn_res_norm"}
    ),
    "vllm/models/kimi_k3/amd/model.py": frozenset({"vision_tower", "mm_projector"}),
}

#: ``finalize_down_hidden`` definition sites: the VL and text-only
#: architectures, in both the nvidia and amd implementations. The ascend
#: classes inherit the amd pair, so a missing amd definition would silently
#: disable the hook on NPU (the only platform LWD targets).
_K3_HOOK_SITES_BY_FILE: dict[str, frozenset[str]] = {
    "vllm/models/kimi_k3/nvidia/model.py": frozenset(
        {"KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"}
    ),
    "vllm/models/kimi_k3/amd/linear.py": frozenset({"KimiLinearForCausalLM"}),
    "vllm/models/kimi_k3/amd/model.py": frozenset({"KimiK3ForConditionalGeneration"}),
}


def _parse_k3_source(rel_path: str) -> ast.Module:
    return ast.parse((_K3_REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _k3_lwd_guard_labels(rel_path: str) -> set[str]:
    """Literal ``lwd_skips_module`` labels used in one K3 source file."""
    labels: set[str] = set()
    for node in ast.walk(_parse_k3_source(rel_path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "lwd_skips_module":
            assert node.args, f"{rel_path}: lwd_skips_module() needs a module path"
            first = node.args[0]
            assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
                f"{rel_path}: LWD guards must pass a literal ownership label, "
                f"got {ast.dump(first)[:60]}"
            )
            labels.add(first.value)
    return labels


@pytest.mark.parametrize("rel_path", sorted(_K3_GUARD_LABELS_BY_FILE))
def test_kimi_k3_guards_match_ownership_table(rel_path: str) -> None:
    """Each K3 implementation guards exactly the modules the table owns."""
    labels = _k3_lwd_guard_labels(rel_path)
    assert labels == _K3_GUARD_LABELS_BY_FILE[rel_path], rel_path
    unknown = labels - LWD_MODULE_OWNERSHIP.keys()
    assert not unknown, f"{rel_path}: labels missing from LWD_MODULE_OWNERSHIP"


@pytest.mark.parametrize("rel_path", sorted(_K3_HOOK_SITES_BY_FILE))
def test_kimi_k3_down_norm_hook_sites(rel_path: str) -> None:
    """The DOWN-norm hook is defined on both architectures, both vendors."""
    owners: set[str] = set()
    for node in ast.walk(_parse_k3_source(rel_path)):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == (
                    "finalize_down_hidden"
                ):
                    owners.add(node.name)
    assert owners == _K3_HOOK_SITES_BY_FILE[rel_path], rel_path
