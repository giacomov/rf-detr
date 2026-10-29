# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vendored C-RADIOv3-B (ViT-B/16) encoder for RF-DETR.

Loads NVIDIA's ``nvidia/C-RADIOv3-B`` weights from a pinned Hugging Face revision into a minimal, timm-free
Vision Transformer that satisfies the ``Backbone._build_encoder`` contract: map ``(B, 3, H, W)`` (ImageNet-
normalized, as RF-DETR always feeds) to a list of ``(B, 768, H/16, W/16)`` feature maps and expose
``_out_feature_channels``.

Architecture facts verified against revision ``44653a0482cf460bb4f12595fc3cc3dfecc403d1``:

- ViT-B/16, dim 768, depth 12, 12 heads, MLP width 3072, qkv bias, LayerScale init ``1e-5``.
- CPE position grid ``128×128`` (``cpe_max_size=2048``); eval path interpolates to ``max(h, w)`` then crops top-left.
- 8 prefix tokens (1 CLS + 7 registers packed into ``cls_token.token``); stripped before returning NCHW maps.
- Checkpoint LayerScale params are named ``grandma`` (HF-safe rename of ``gamma``); remapped on load.
- ``input_conditioner`` expects ``[0, 1]`` RGB with OpenAI CLIP mean/std; we fold ImageNet→CLIP into one affine so
  RF-DETR's ImageNet-normalized inputs need no public-API preprocessing change.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Pinned Hugging Face Hub repo and revision for C-RADIOv3-B weights.
HUB_REPO_ID = "nvidia/C-RADIOv3-B"
HUB_REVISION = "44653a0482cf460bb4f12595fc3cc3dfecc403d1"
WEIGHTS_FILENAME = "model.safetensors"

EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4.0
PATCH_SIZE = 16
NUM_PREFIX_TOKENS = 8
CPE_GRID_SIZE = 128  # cpe_max_size // patch_size == 2048 // 16
LAYER_SCALE_INIT = 1e-5

#: ImageNet stats used by RF-DETR's public preprocessing.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
#: OpenAI CLIP stats used by RADIO's ``input_conditioner`` (``timm.data.constants``).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _imagenet_to_clip_affine() -> tuple[Tensor, Tensor]:
    """Return ``(scale, shift)`` such that ``y = x * scale + shift`` maps ImageNet-normalized ``x`` to CLIP space.

    Derivation: pixel = x * im_std + im_mean, then y = (pixel - clip_mean) / clip_std.

    Examples:
        >>> scale, shift = _imagenet_to_clip_affine()
        >>> scale.shape
        torch.Size([1, 3, 1, 1])
        >>> # Identity-ish: a zero ImageNet tensor is mean-grey, not CLIP-zero.
        >>> float(shift.abs().sum()) > 0
        True
    """
    im_mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    im_std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    clip_mean = torch.tensor(_CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    clip_std = torch.tensor(_CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
    scale = im_std / clip_std
    shift = (im_mean - clip_mean) / clip_std
    return scale, shift


def _cpe_pos_embed_eval(pos_embed: Tensor, height_tokens: int, width_tokens: int) -> Tensor:
    """Reproduce RADIO's eval CPE path: interpolate to ``max(h,w)²``, crop top-left to ``(h, w)``.

    Args:
        pos_embed: Absolute position grid of shape ``(1, CPE_GRID_SIZE * CPE_GRID_SIZE, C)``.
        height_tokens: Target height in tokens.
        width_tokens: Target width in tokens.

    Returns:
        Position embeddings of shape ``(1, height_tokens * width_tokens, C)``.

    Examples:
        >>> pe = torch.randn(1, CPE_GRID_SIZE * CPE_GRID_SIZE, 8)
        >>> out = _cpe_pos_embed_eval(pe, 4, 6)
        >>> tuple(out.shape)
        (1, 24, 8)
    """
    if height_tokens == CPE_GRID_SIZE and width_tokens == CPE_GRID_SIZE:
        return pos_embed
    # (1, C, rows, cols)
    grid = pos_embed.reshape(1, CPE_GRID_SIZE, CPE_GRID_SIZE, -1).permute(0, 3, 1, 2)
    max_dim = max(height_tokens, width_tokens)
    # align_corners=True matches the Hugging Face Hub copy of vit_patch_generator.py at the pinned revision
    # (GitHub main later flipped this to False; keep True for weight/feature parity with the Hub checkpoint).
    if (grid.shape[-2], grid.shape[-1]) != (max_dim, max_dim):
        grid = F.interpolate(grid.float(), size=(max_dim, max_dim), mode="bilinear", align_corners=True).to(
            pos_embed.dtype
        )
    grid = grid[..., :height_tokens, :width_tokens]
    if grid.shape[-2:] != (height_tokens, width_tokens):
        grid = F.interpolate(grid.float(), size=(height_tokens, width_tokens), mode="bilinear", align_corners=True).to(
            pos_embed.dtype
        )
    return grid.flatten(2).permute(0, 2, 1)


class DropPath(nn.Module):
    """Stochastic depth; exposes ``drop_prob`` so ``LWDETR.update_drop_path`` can rewrite the schedule."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth when training and ``drop_prob > 0``."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        if keep > 0.0:
            mask = mask.div_(keep)
        return x * mask


class LayerScale(nn.Module):
    """Per-channel residual scale (DINOv2 / RADIO LayerScale) with param name ``gamma``."""

    def __init__(self, dim: int, init_values: float = LAYER_SCALE_INIT) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Scale ``x`` by the learned per-channel ``gamma``."""
        return x * self.gamma


class Attention(nn.Module):
    """Multi-head self-attention matching timm's classic ``qkv`` Linear layout."""

    def __init__(self, dim: int, num_heads: int = NUM_HEADS, qkv_bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Self-attend over the token sequence."""
        batch, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        x = F.scaled_dot_product_attention(query, key, value)
        x = x.transpose(1, 2).reshape(batch, num_tokens, channels)
        return cast(Tensor, self.proj(x))


class Mlp(nn.Module):
    """Two-layer GELU MLP matching timm's ``fc1``/``fc2`` naming (no dropout; RADIO trains with drop=0)."""

    def __init__(self, dim: int, mlp_ratio: float = MLP_RATIO) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network."""
        return cast(Tensor, self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm ViT block with LayerScale and a shared DropPath (both residuals)."""

    def __init__(self, dim: int = EMBED_DIM, num_heads: int = NUM_HEADS, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
        self.ls2 = LayerScale(dim)
        # One module shared by both residuals so ``update_drop_path`` (which sets ``block.drop_path.drop_prob``)
        # rewrites both branches. Also aliased as drop_path1/drop_path2 to mirror timm naming.
        self.drop_path = DropPath(drop_path)
        self.drop_path1 = self.drop_path
        self.drop_path2 = self.drop_path

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention and MLP residuals."""
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def remap_c_radio_v3_state_dict(raw: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map a Hugging Face ``model.safetensors`` state dict onto :class:`CRadioV3` parameter names.

    Args:
        raw: Checkpoint tensors keyed as ``radio_model.*``.

    Returns:
        Remapped state dict ready for ``load_state_dict(..., strict=True)``.

    Raises:
        KeyError: If a required checkpoint key is missing.
        ValueError: If a tensor has an unexpected shape.

    Examples:
        >>> w = torch.randn(EMBED_DIM, 3 * PATCH_SIZE * PATCH_SIZE)
        >>> pe = torch.randn(1, CPE_GRID_SIZE * CPE_GRID_SIZE, EMBED_DIM)
        >>> pref = torch.randn(NUM_PREFIX_TOKENS, EMBED_DIM)
        >>> raw = {
        ...     "radio_model.model.patch_generator.embedder.weight": w,
        ...     "radio_model.model.patch_generator.pos_embed": pe,
        ...     "radio_model.model.patch_generator.cls_token.token": pref,
        ...     "radio_model.model.blocks.0.attn.qkv.weight": torch.randn(EMBED_DIM * 3, EMBED_DIM),
        ...     "radio_model.model.blocks.0.attn.qkv.bias": torch.randn(EMBED_DIM * 3),
        ...     "radio_model.model.blocks.0.attn.proj.weight": torch.randn(EMBED_DIM, EMBED_DIM),
        ...     "radio_model.model.blocks.0.attn.proj.bias": torch.randn(EMBED_DIM),
        ...     "radio_model.model.blocks.0.norm1.weight": torch.ones(EMBED_DIM),
        ...     "radio_model.model.blocks.0.norm1.bias": torch.zeros(EMBED_DIM),
        ...     "radio_model.model.blocks.0.norm2.weight": torch.ones(EMBED_DIM),
        ...     "radio_model.model.blocks.0.norm2.bias": torch.zeros(EMBED_DIM),
        ...     "radio_model.model.blocks.0.mlp.fc1.weight": torch.randn(int(EMBED_DIM * MLP_RATIO), EMBED_DIM),
        ...     "radio_model.model.blocks.0.mlp.fc1.bias": torch.randn(int(EMBED_DIM * MLP_RATIO)),
        ...     "radio_model.model.blocks.0.mlp.fc2.weight": torch.randn(EMBED_DIM, int(EMBED_DIM * MLP_RATIO)),
        ...     "radio_model.model.blocks.0.mlp.fc2.bias": torch.randn(EMBED_DIM),
        ...     "radio_model.model.blocks.0.ls1.grandma": torch.ones(EMBED_DIM) * LAYER_SCALE_INIT,
        ...     "radio_model.model.blocks.0.ls2.grandma": torch.ones(EMBED_DIM) * LAYER_SCALE_INIT,
        ...     "radio_model.model.reg_token": torch.randn(1, 4, EMBED_DIM),
        ...     "radio_model.summary_idxs": torch.tensor([0, 1, 2], dtype=torch.int64),
        ...     "radio_model.input_conditioner.norm_mean": torch.tensor(_CLIP_MEAN).view(3, 1, 1),
        ...     "radio_model.input_conditioner.norm_std": torch.tensor(_CLIP_STD).view(3, 1, 1),
        ... }
        >>> out = remap_c_radio_v3_state_dict(raw)
        >>> out["patch_embed.weight"].shape
        torch.Size([768, 3, 16, 16])
        >>> "blocks.0.ls1.gamma" in out and "reg_token" not in out
        True
    """
    prefix = "radio_model.model."
    conditioner_mean_key = "radio_model.input_conditioner.norm_mean"
    conditioner_std_key = "radio_model.input_conditioner.norm_std"
    drop_exact = {
        "radio_model.model.reg_token",
        "radio_model.summary_idxs",
        conditioner_mean_key,
        conditioner_std_key,
    }
    out: dict[str, Tensor] = {}
    for key, value in raw.items():
        if key in drop_exact or key.startswith("radio_model.adaptor") or key.endswith("feature_normalizer"):
            continue
        if not key.startswith(prefix):
            # Ignore other top-level RADIO buffers we do not consume.
            continue
        local = key[len(prefix) :]
        if local == "patch_generator.embedder.weight":
            if value.ndim != 2 or value.shape != (EMBED_DIM, 3 * PATCH_SIZE * PATCH_SIZE):
                raise ValueError(f"Unexpected patch embedder weight shape {tuple(value.shape)}.")
            # Linear over (c, yy, xx) == Conv2d(3, embed, patch, patch) weight layout.
            out["patch_embed.weight"] = value.view(EMBED_DIM, 3, PATCH_SIZE, PATCH_SIZE).clone()
            continue
        if local == "patch_generator.pos_embed":
            out["pos_embed"] = value
            continue
        if local == "patch_generator.cls_token.token":
            out["prefix_tokens"] = value
            continue
        if local.endswith(".grandma"):
            local = local[: -len("grandma")] + "gamma"
        out[local] = value

    required = ("patch_embed.weight", "pos_embed", "prefix_tokens")
    missing = [name for name in required if name not in out]
    if missing:
        raise KeyError(f"C-RADIOv3-B checkpoint missing required keys after remap: {missing}.")
    return out


def load_c_radio_v3_weights(model: CRadioV3) -> None:
    """Download the pinned Hub checkpoint and load it into ``model`` with ``strict=True``.

    Args:
        model: A randomly initialized :class:`CRadioV3`.

    Raises:
        RuntimeError: If the Hub download or safetensors load fails.
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "Loading C-RADIOv3-B weights requires huggingface_hub and safetensors "
            "(both ship with transformers, which rfdetr already depends on)."
        ) from exc

    try:
        path = hf_hub_download(HUB_REPO_ID, WEIGHTS_FILENAME, revision=HUB_REVISION)
        raw = load_file(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download C-RADIOv3-B weights from {HUB_REPO_ID}@{HUB_REVISION}. "
            "Check network access and that the NVIDIA Open Model License terms on the Hub page are acceptable."
        ) from exc

    remapped = remap_c_radio_v3_state_dict(dict(raw))
    # Prefer the checkpoint's own CLIP mean/std for the ImageNet→CLIP affine when present.
    mean_key = "radio_model.input_conditioner.norm_mean"
    std_key = "radio_model.input_conditioner.norm_std"
    if mean_key in raw and std_key in raw:
        clip_mean = raw[mean_key].detach().float().view(1, 3, 1, 1)
        clip_std = raw[std_key].detach().float().view(1, 3, 1, 1)
        im_mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        im_std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        model.in_scale.copy_(im_std / clip_std)
        model.in_shift.copy_((im_mean - clip_mean) / clip_std)

    incompatible = model.load_state_dict(remapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        # strict=True already raises on mismatch; this branch is defensive for older torch stubs.
        raise RuntimeError(f"C-RADIOv3-B weight load mismatch: {incompatible}.")


class CRadioV3(nn.Module):
    """Vendored C-RADIOv3-B encoder returning multi-depth NCHW feature maps at stride 16."""

    def __init__(
        self,
        out_feature_indexes: list[int],
        *,
        target_shape: tuple[int, int] = (512, 512),
        load_pretrained_weights: bool = True,
        drop_path_rate: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not out_feature_indexes:
            raise ValueError("out_feature_indexes must be a non-empty list of 1-based block indices.")
        if any(i < 1 or i > DEPTH for i in out_feature_indexes):
            raise ValueError(
                f"out_feature_indexes must be in 1..{DEPTH} (1-based block indices), got {out_feature_indexes}."
            )
        if sorted(out_feature_indexes) != list(out_feature_indexes):
            raise ValueError(f"out_feature_indexes must be ascending, got {out_feature_indexes}.")

        self.patch_size = PATCH_SIZE
        self.out_feature_indexes = list(out_feature_indexes)
        self.gradient_checkpointing = gradient_checkpointing
        self._out_feature_channels: list[int] = [EMBED_DIM] * len(out_feature_indexes)

        self.patch_embed = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, CPE_GRID_SIZE * CPE_GRID_SIZE, EMBED_DIM))
        self.prefix_tokens = nn.Parameter(torch.zeros(NUM_PREFIX_TOKENS, EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.prefix_tokens, std=0.02)

        # Linear drop-path schedule across depth (timm default); LWDETR.update_drop_path may rewrite later.
        dpr = [drop_path_rate * i / max(DEPTH - 1, 1) for i in range(DEPTH)] if drop_path_rate > 0 else [0.0] * DEPTH
        self.blocks = nn.ModuleList([Block(drop_path=rate) for rate in dpr])

        scale, shift = _imagenet_to_clip_affine()
        self.in_scale: Tensor
        self.in_shift: Tensor
        self.register_buffer("in_scale", scale, persistent=False)
        self.register_buffer("in_shift", shift, persistent=False)

        # Baked position embeddings for the native / export resolution (avoids interpolate in the ONNX graph).
        self._target_shape = target_shape
        self._baked_token_hw: tuple[int, int] | None = None
        self.baked_pos_embed: Tensor
        self.register_buffer("baked_pos_embed", torch.empty(0), persistent=False)

        if load_pretrained_weights:
            load_c_radio_v3_weights(self)
        # Bake after weight load so the buffer reflects the pretrained CPE grid (not the random init).
        self.set_export_shape(target_shape)

    def set_export_shape(self, shape: tuple[int, int]) -> None:
        """Re-bake CPE position embeddings for ``shape`` so export traces avoid runtime interpolate.

        Export preparation may call this more than once with different shapes; each call replaces the buffer.

        Args:
            shape: Input ``(height, width)`` in pixels. Both sides must be divisible by ``patch_size``.
        """
        height, width = shape
        if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
            raise ValueError(f"C-RADIOv3-B requires height and width divisible by {PATCH_SIZE}, got shape={shape}.")
        token_h, token_w = height // PATCH_SIZE, width // PATCH_SIZE
        baked = _cpe_pos_embed_eval(self.pos_embed.detach(), token_h, token_w)
        self._baked_token_hw = (token_h, token_w)
        # Buffer must live on the same device/dtype as parameters for export tracing.
        self.register_buffer("baked_pos_embed", baked.to(device=self.pos_embed.device, dtype=self.pos_embed.dtype))

    def _pos_embed_for(self, token_h: int, token_w: int) -> Tensor:
        """Return position embeddings for a ``token_h × token_w`` grid, preferring the baked buffer."""
        if self._baked_token_hw == (token_h, token_w) and self.baked_pos_embed.numel() > 0:
            return self.baked_pos_embed
        return _cpe_pos_embed_eval(self.pos_embed, token_h, token_w)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return one NCHW feature map per ``out_feature_indexes`` entry (finest depth order preserved)."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input (B, 3, H, W), got {tuple(x.shape)}.")
        if x.shape[-2] % PATCH_SIZE != 0 or x.shape[-1] % PATCH_SIZE != 0:
            raise ValueError(f"Spatial dims must be divisible by {PATCH_SIZE}, got H={x.shape[-2]}, W={x.shape[-1]}.")

        x = x * self.in_scale + self.in_shift
        tokens = self.patch_embed(x)  # (B, C, Ht, Wt)
        batch, channels, token_h, token_w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, Ht*Wt, C)
        tokens = tokens + self._pos_embed_for(token_h, token_w)
        prefix = self.prefix_tokens.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat([prefix, tokens], dim=1)

        wanted = {idx: position for position, idx in enumerate(self.out_feature_indexes)}
        outputs: list[Tensor | None] = [None] * len(self.out_feature_indexes)
        for block_idx, block in enumerate(self.blocks, start=1):
            if self.gradient_checkpointing and self.training:
                tokens = cast(Tensor, checkpoint(block, tokens, use_reentrant=False))
            else:
                tokens = block(tokens)
            if block_idx in wanted:
                spatial = tokens[:, NUM_PREFIX_TOKENS:, :].transpose(1, 2).reshape(batch, channels, token_h, token_w)
                outputs[wanted[block_idx]] = spatial

        assert all(feat is not None for feat in outputs)
        return [cast(Tensor, feat) for feat in outputs]
