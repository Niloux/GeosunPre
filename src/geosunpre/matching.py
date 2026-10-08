"""Pinned LoMa inference without unused math-SDPA outputs or reverse matches."""
from types import MethodType


def flash_self_attention(self, x, encoding):
    import torch
    from loma.loma import apply_cached_rotary_emb
    from torch.nn.attention import SDPBackend, sdpa_kernel
    qkv = self.Wqkv(x).unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
    q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
    q = apply_cached_rotary_emb(encoding, q)
    k = apply_cached_rotary_emb(encoding, k)
    # Require the fixed production kernel; fail if the GPU cannot run it.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        context = torch.nn.functional.scaled_dot_product_attention(q, k, v.contiguous())
    message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
    return x + self.ffn(torch.cat([x, message], -1))


def configure_matcher(model):
    from loma.loma import SelfBlock
    for layer in model.transformers:
        if not isinstance(layer.self_attn, SelfBlock):
            raise TypeError('Unsupported LoMa self-attention implementation')
        layer.self_attn.forward = MethodType(flash_self_attention, layer.self_attn)


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
