"""Pinned LoMa inference without unused math-SDPA outputs or reverse matches."""
import os
from types import MethodType


def math_self_attention(self, x, encoding):
    import torch
    from loma.loma import apply_cached_rotary_emb
    qkv = self.Wqkv(x).unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
    q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
    q = apply_cached_rotary_emb(encoding, q)
    k = apply_cached_rotary_emb(encoding, k)
    # Match torch 2.8 math SDPA: scale both operands, accumulate in float32,
    # and cast only the context back. Do not materialize the unused attention
    # weights in the original dtype. No attention mask or dropout is used here.
    with torch.autocast(device_type=x.device.type, enabled=False):
        scale = self.head_dim ** -.25
        scores = (q.float() * scale) @ (k.float().transpose(-2, -1) * scale)
        probabilities = scores.softmax(-1)
        # Equivalent to _safe_softmax, including fully masked (-inf) rows,
        # without allocating a Boolean tensor the size of the score matrix.
        probabilities.masked_fill_(torch.isneginf(scores.amax(-1, keepdim=True)), 0.)
        context = (probabilities @ v.float()).to(q.dtype)
    message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
    return x + self.ffn(torch.cat([x, message], -1))


def flash_self_attention(self, x, encoding):
    import torch
    from loma.loma import apply_cached_rotary_emb
    from torch.nn.attention import SDPBackend, sdpa_kernel
    qkv = self.Wqkv(x).unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
    q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
    q = apply_cached_rotary_emb(encoding, q)
    k = apply_cached_rotary_emb(encoding, k)
    # Explicit opt-in: flash changes rounding and therefore some matches.
    # Require the requested kernel rather than silently timing another backend.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        context = torch.nn.functional.scaled_dot_product_attention(q, k, v.contiguous())
    message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
    return x + self.ffn(torch.cat([x, message], -1))


def configure_matcher(model, attention='flash'):
    from loma.loma import SelfBlock
    if attention not in ('math', 'flash'):
        raise ValueError('Unsupported matcher attention')
    if attention == 'math' and os.environ.get('GEOSUNPRE_MATCH_MATH', 'optimized') == 'reference':
        return
    for layer in model.transformers:
        if not isinstance(layer.self_attn, SelfBlock):
            raise TypeError('Unsupported LoMa self-attention implementation')
        implementation = flash_self_attention if attention == 'flash' else math_self_attention
        layer.self_attn.forward = MethodType(implementation, layer.self_attn)


def forward_matches(scores, threshold):
    """The used half of LoMa filter_matches, with the same mutual-nearest rule."""
    import torch
    maximum, indices = scores.max(2)
    reverse = scores.max(1).indices
    mutual = torch.arange(indices.shape[1], device=indices.device)[None] == reverse.gather(1, indices)
    confidence = torch.where(mutual, maximum, 0.)
    return torch.where(mutual & (confidence > threshold), indices, -1), confidence


class MatcherGraph:
    """One bounded CUDA graph for the common shape; eager fallback for other sizes."""
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        self.graph = None
        self.shape = None

    def eager(self, *inputs):
        return forward_matches(self.model(*inputs)['scores'], self.threshold)

    def __call__(self, *inputs):
        import torch
        shape = tuple((tuple(t.shape), t.dtype, t.device) for t in inputs)
        if os.environ.get('GEOSUNPRE_MATCH_GRAPH', '1') == '0':
            return self.eager(*inputs)
        if self.graph is None:
            self.shape = shape
            self.inputs = tuple(t.clone() for t in inputs)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.eager(*self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self.eager(*self.inputs)
        if shape != self.shape:
            return self.eager(*inputs)
        for destination, value in zip(self.inputs, inputs):
            destination.copy_(value)
        self.graph.replay()
        return self.outputs
