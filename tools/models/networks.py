"""Adapters for the pinned LoMa, MoGe and SAM3 native tensor contracts."""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class Detector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        # DaD owns ImageNet normalization; callers supply RGB / 255.
        return self.model.forward_impl(image).float()


class Descriptor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, keypoints):
        dense = self.model(image).float()
        return F.grid_sample(dense, keypoints[:, None], mode='bilinear',
                             align_corners=False)[:, :, 0].transpose(1, 2).float()


def fix_descriptor_positions(model):
    vit = model.encoder.frozen_dinov2.dinov2_vitl14
    with torch.inference_mode():
        image = torch.zeros((1, 3, 784, 784), device='cuda')
        patches = vit.patch_embed(image)
        tokens = torch.cat((vit.cls_token.expand(1, -1, -1), patches), dim=1)
        vit.register_buffer('_export_positions', vit.interpolate_pos_encoding(tokens, 784, 784))

    def positions(self, x, width, height):
        return self._export_positions.to(x.dtype)

    vit.interpolate_pos_encoding = MethodType(positions, vit)

    # The upstream compiler-disable decorator cannot be traced by ONNX.
    def frozen_forward(self, image):
        features = self.dinov2_vitl14.forward_features(image)['x_norm_patchtokens']
        return [features.transpose(1, 2).reshape(1, 1024, 56, 56)], [(56, 56)]

    model.encoder.frozen_dinov2.forward = MethodType(frozen_forward, model.encoder.frozen_dinov2)


class Matcher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def attention(q, k, v, mask):
        similarity = (q @ k.transpose(2, 3)) * (q.shape[3] ** -.5)
        similarity = torch.where(mask[:, None, None, :] > 0, similarity, -10000.)
        return F.softmax(similarity, dim=3) @ v

    def encoding(self, keypoints):
        projected = self.model.posenc.Wr(keypoints)
        cosine = torch.stack((projected.cos(), projected.cos()), dim=3).reshape(1, -1, 64)
        sine = torch.stack((projected.sin(), projected.sin()), dim=3).reshape(1, -1, 64)
        return cosine[:, None], sine[:, None]

    def self_attention(self, block, x, encoding, mask):
        qkv = block.Wqkv(x).reshape(1, -1, block.num_heads, block.head_dim, 3).transpose(1, 2)
        def rotary(tensor):
            # Explicit pairs avoid the legacy ONNX exporter's negative-axis unflatten.
            rotated = torch.stack((-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), dim=4)
            rotated = rotated.reshape(1, block.num_heads, -1, block.head_dim)
            return tensor * encoding[0] + rotated * encoding[1]
        q = rotary(qkv.select(4, 0))
        k = rotary(qkv.select(4, 1))
        context = self.attention(q, k, qkv.select(4, 2), mask)
        message = block.out_proj(context.transpose(1, 2).reshape(1, -1, block.embed_dim))
        return x + block.ffn(torch.cat((x, message), 2))

    def cross_attention(self, block, a, b, ma, mb):
        qka, qkb = block.to_qk(a), block.to_qk(b)
        va, vb = block.to_v(a), block.to_v(b)
        qka, qkb, va, vb = [t.reshape(1, -1, block.heads, t.shape[-1] // block.heads).transpose(1, 2)
                             for t in (qka, qkb, va, vb)]
        xa = block.to_out(self.attention(qka, qkb, vb, mb).transpose(1, 2).reshape(1, -1, block.to_out.in_features))
        xb = block.to_out(self.attention(qkb, qka, va, ma).transpose(1, 2).reshape(1, -1, block.to_out.in_features))
        return (a + block.ffn(torch.cat((a, xa), 2)),
                b + block.ffn(torch.cat((b, xb), 2)))

    def forward(self, ka, kb, da, db, mask_a, mask_b):
        a, b = self.model.input_proj(da), self.model.input_proj(db)
        ea, eb = self.encoding(ka), self.encoding(kb)
        for layer in self.model.transformers:
            a = self.self_attention(layer.self_attn, a, ea, mask_a)
            b = self.self_attention(layer.self_attn, b, eb, mask_b)
            a, b = self.cross_attention(layer.cross_attn, a, b, mask_a, mask_b)
        final = self.model.log_assignment[-1].final_proj
        a, b = final(a) / 4., final(b) / 4.  # 256 ** 0.25
        similarity = a @ b.transpose(1, 2)
        pa = F.softmax(torch.where(mask_b[:, None, :] > 0, similarity, -10000.), dim=2)
        pb = F.softmax(torch.where(mask_a[:, :, None] > 0, similarity, -10000.), dim=1)
        scores = pa * pb * mask_a[:, :, None] * mask_b[:, None, :]
        confidence, index = scores.max(2)
        reverse = scores.argmax(1)
        mutual = reverse.gather(1, index) == torch.arange(ka.shape[1], device=ka.device)[None]
        valid = mutual & (confidence > .1) & (mask_a > 0) & (mask_b.gather(1, index) > 0)
        return torch.where(valid, index, -1).to(torch.int32), torch.where(mutual, confidence, 0.).float()


def fix_geometry_positions(model, size):
    """Freeze upstream DINOv2 positions at the fixed input before ONNX mode changes their interpolation."""
    vit = model.encoder.backbone
    with torch.inference_mode():
        tokens = torch.zeros((1, (size // 14) ** 2 + 1, vit.embed_dim), device='cuda')
        vit.register_buffer('_export_positions', vit.interpolate_pos_encoding(tokens, size, size).clone())

    def positions(self, x, width, height):
        return self._export_positions.to(x.dtype)

    vit.interpolate_pos_encoding = MethodType(positions, vit)
    model.onnx_compatible_mode = True


class Geometry(nn.Module):
    """MoGe-2 raw forward at a fixed square size and token count; post-processing stays native."""
    def __init__(self, model, tokens):
        super().__init__()
        self.model = model
        self.tokens = tokens

    def forward(self, image):
        output = self.model(image, self.tokens)
        return tuple(output[key].float() for key in ('points', 'normal', 'mask', 'metric_scale'))


def real_rotary(xq, xk, freqs_cis, repeat_freqs_k=False):
    """Upstream rotary encoding on (real, imaginary) pairs; ONNX has no complex tensors."""
    cos, sin = freqs_cis.real.to(xq.dtype), freqs_cis.imag.to(xq.dtype)

    def rotate(x, c, s):
        x = x.unflatten(-1, (-1, 2))
        a, b = x[..., 0], x[..., 1]
        return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)

    if xk.shape[-2] == 0:
        return rotate(xq, cos, sin), xk
    c, s = cos, sin
    if repeat_freqs_k and xk.shape[-2] != xq.shape[-2]:
        r = xk.shape[-2] // xq.shape[-2]
        c, s = c.repeat(*([1] * (c.ndim - 2)), r, 1), s.repeat(*([1] * (s.ndim - 2)), r, 1)
    return rotate(xq, cos, sin), rotate(xk, c, s)


# TensorRT keeps SAM3 attention logits in FP32 and so cannot fuse attention: each global attention
# over the 72x72 image tokens materializes its full logit square (3.3 GB in the DETR encoder).
# Query blocks of this size bound that scratch to one block of rows.
ATTENTION_BLOCK = 648


def blocked_attention(q, k, v):
    """softmax(q k^T / sqrt(d)) v computed over query blocks."""
    k = k.transpose(-2, -1) * q.shape[-1] ** -.5

    def rows(block):
        return torch.softmax(block @ k, -1) @ v

    if q.shape[-2] <= ATTENTION_BLOCK:
        return rows(q)
    return torch.cat([rows(block) for block in q.split(ATTENTION_BLOCK, -2)], -2)


def blocked_sdpa(q, k, v, **options):
    if options:
        raise RuntimeError(f'Blocked attention supports plain attention only: {sorted(options)}')
    return blocked_attention(q, k, v)


class BlockedSelfAttention(nn.Module):
    """nn.MultiheadAttention without masks, through blocked attention."""
    def __init__(self, attention):
        super().__init__()
        if (not attention._qkv_same_embed_dim or attention.in_proj_bias is None or
                attention.bias_k is not None):
            raise RuntimeError('Blocked attention expects packed self-attention with input bias')
        self.attention = attention

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True, **_):
        if attn_mask is not None or key_padding_mask is not None:
            raise RuntimeError('Blocked attention does not support attention masks')
        a = self.attention
        w, b = a.in_proj_weight.chunk(3), a.in_proj_bias.chunk(3)

        # (batch, heads, length, channels) from batch-first or sequence-first inputs.
        def split(x, i):
            x = F.linear(x if a.batch_first else x.transpose(0, 1), w[i], b[i])
            return x.unflatten(-1, (a.num_heads, -1)).transpose(1, 2)

        x = a.out_proj(blocked_attention(split(query, 0), split(key, 1), split(value, 2))
                       .transpose(1, 2).flatten(2))
        return (x if a.batch_first else x.transpose(0, 1)), None


def block_grounding_attention(model, vitdet, encoder):
    """Route the SAM3 trunk and DETR encoder self-attention through blocked attention."""
    class Functional:
        scaled_dot_product_attention = staticmethod(blocked_sdpa)

        def __getattr__(self, name):
            return getattr(F, name)

    vitdet.F = Functional()
    layers = [m for m in model.modules() if isinstance(m, encoder.TransformerEncoderLayer)
              and isinstance(m.self_attn, nn.MultiheadAttention)]
    if not layers:
        raise RuntimeError('SAM3 has no encoder self-attention to block')
    for layer in layers:
        layer.self_attn = BlockedSelfAttention(layer.self_attn)


class Grounding(nn.Module):
    """SAM3 concept detection for fixed prompts whose text embeddings are baked into the graph.

    Per prompt: the top-k joint scores (query score times presence), their normalized cxcywh
    boxes and mask logits, and the semantic logits of the prompt."""
    def __init__(self, model, prompts, top_k):
        super().__init__()
        self.model = model
        self.top_k = top_k
        self.register_buffer('text_ids', torch.arange(len(prompts)))

    def forward(self, image):
        features = self.model.backbone.forward_image(image)
        features.pop('sam2_backbone_out', None)
        out = self.model.forward_grounding(features, self.text_ids)
        scores = out['pred_logits'].sigmoid().squeeze(-1) * out['presence_logit_dec'].sigmoid()
        scores, index = scores.topk(self.top_k, dim=1)
        boxes = out['pred_boxes'].gather(1, index[..., None].expand(-1, -1, 4))
        masks = out['pred_masks']
        masks = masks.gather(1, index[..., None, None].expand(-1, -1, *masks.shape[-2:]))
        return scores.float(), boxes.float(), masks.float(), out['semantic_seg'].float()
