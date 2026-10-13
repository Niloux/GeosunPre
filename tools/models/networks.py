"""Adapters for the pinned LoMa and SegFormer native tensor contracts."""
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


class Segmentation(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        labels = {v: int(k) for k, v in model.config.id2label.items()}
        self.labels = [labels['sky'], labels['person']]

    def forward(self, image):
        probability = self.model(image, return_dict=False)[0].softmax(1)[:, self.labels]
        return F.interpolate(probability, (512, 512), mode='bilinear', align_corners=False).float()
