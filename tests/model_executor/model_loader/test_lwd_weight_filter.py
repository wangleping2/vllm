# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LWD weight filtering during model loading."""

from functools import partial

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.default_loader import _skip_unowned_weights
from vllm.model_executor.model_loader.lwd_weight_filter import (
    lwd_should_skip_weight,
    parse_skip_keys,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)


class TestParseSkipKeys:
    def test_edge_skips_cloud_owned_modules(self):
        assert parse_skip_keys("edge") == frozenset(
            {"model.layers", "model.norm", "model.mtp"}
        )

    def test_cloud_skips_edge_owned_modules(self):
        assert parse_skip_keys("cloud") == frozenset(
            {"visual", "vision_tower", "mm_projector"}
        )


class TestLwdShouldSkipWeightEdge:
    """The edge owns embed/lm_head/visual and skips layers/norm/mtp."""

    @pytest.mark.parametrize(
        "name",
        [
            # flat HF names
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
            # DeepSeek MTP arrives as an extra layer in the HF checkpoint and
            # dies with the layers subtree (tid2eid included)
            "model.layers.61.mlp.gate.tid2eid.weight",
            # VL checkpoints (Qwen3-VL / GLM-4.5V) nest the language model
            "model.language_model.norm.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            # vLLM module-tree form: pins the defensive nested matching
            "model.language_model.model.norm.weight",
            # native DeepSeek layout without the leading "model"
            "layers.3.mlp.w1.weight",
            "norm.weight",
            # Kimi-K3 layout: no leading "model." on the VL nesting
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.norm.weight",
            # K3's attention-residual tail is cloud-owned
            "language_model.model.output_attn_res_norm.weight",
            "language_model.model.output_attn_res_proj.weight",
        ],
    )
    def test_skips(self, name):
        assert lwd_should_skip_weight(name, "edge")

    @pytest.mark.parametrize(
        "name",
        [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "language_model.lm_head.weight",
            # tower-internal norms must survive: the tower dies via its root
            "visual.merger.norm.weight",
            "model.visual.merger.norm.weight",
            "model.visual.blocks.0.norm1.weight",
            # dot boundaries: not fooled by longer leaf names
            "model.norm_eps.weight",
            "visualizer.weight",
            # Kimi-K3 tower/projector belong to the edge
            "vision_tower.encoder.blocks.0.attn.qkv.weight",
            "vision_tower.encoder.blocks.0.norm1.weight",
            "mm_projector.linear_1.weight",
            "mm_projector.proj.2.weight",
        ],
    )
    def test_keeps(self, name):
        assert not lwd_should_skip_weight(name, "edge")


class TestLwdShouldSkipWeightCloud:
    """The cloud owns layers/norm/mtp and skips the visual tower."""

    @pytest.mark.parametrize(
        "name",
        [
            # flat Qwen tower names
            "visual.blocks.0.attn.qkv.weight",
            "visual.merger.mlp.0.weight",
            # nested Qwen multimodal tower names
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.merger.deepstack_merger_list.0.fc.weight",
            # Kimi-K3 names the tower/projector without the "visual" prefix
            "vision_tower.encoder.blocks.0.attn.qkv.weight",
            "mm_projector.linear_1.weight",
        ],
    )
    def test_skips(self, name):
        assert lwd_should_skip_weight(name, "cloud")

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
            "model.language_model.norm.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            # Kimi-K3's final-norm tail is cloud-owned
            "language_model.model.output_attn_res_norm.weight",
            "language_model.model.output_attn_res_proj.weight",
        ],
    )
    def test_keeps(self, name):
        assert not lwd_should_skip_weight(name, "cloud")


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
        "visual.blocks.0.attn.qkv.weight",
    ],
)
def test_non_lwd_role_keeps_everything(name):
    # Non-LWD deployments must keep the pre-LWD loading behavior.
    assert not lwd_should_skip_weight(name, None)


def test_safetensors_iterator_applies_skip_name(tmp_path):
    tensors = {
        "model.embed_tokens.weight": torch.zeros(2),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(2),
        "model.norm.weight": torch.zeros(2),
        "visual.blocks.0.attn.qkv.weight": torch.zeros(2),
    }
    st_file = tmp_path / "model.safetensors"
    save_file(tensors, str(st_file))

    got = {
        name
        for name, _ in safetensors_weights_iterator(
            [str(st_file)],
            use_tqdm_on_load=False,
            skip_name=partial(lwd_should_skip_weight, role="edge"),
        )
    }

    assert got == {"model.embed_tokens.weight", "visual.blocks.0.attn.qkv.weight"}


def test_skip_unowned_weights_passthrough_without_filter():
    weights = [("model.norm.weight", object())]
    assert list(_skip_unowned_weights(iter(weights), None)) == weights


def test_skip_unowned_weights_drops_unowned_tensors():
    weights = [
        ("model.norm.weight", 0),
        ("model.embed_tokens.weight", 1),
    ]

    got = list(
        _skip_unowned_weights(
            iter(weights), partial(lwd_should_skip_weight, role="edge")
        )
    )

    assert got == [("model.embed_tokens.weight", 1)]
