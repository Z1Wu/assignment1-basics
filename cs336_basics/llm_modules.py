import torch
from torch.nn import Module, Parameter, ModuleList
import math
import einops
from jaxtyping import Bool, Float, Int
from torch import Tensor
from typing import Optional, Callable, Iterable



def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    sum_x = torch.sum(x, dim=dim, keepdim=True)
    return x / sum_x

def cross_entropy_loss(logits: Float[Tensor, " batch_size vocab_size"], y: Float[Tensor, " batch_size "]) -> torch.Tensor: 
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    sum_logits = torch.sum(torch.exp(logits), dim=-1, keepdim=True)
    y = einops.rearrange(y, '... batch_size -> ... batch_size 1')
    y_logits = torch.gather(logits, 1, y)
    return torch.mean(torch.log(sum_logits) - y_logits)


def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    dot_product_mat = einops.einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys" 
    ) / math.sqrt(d_k)
    if mask != None:
        dot_product_mat[~mask] = -float('inf')
    prob_mat = softmax(dot_product_mat, dim = -1)
    res = einops.einsum(
        prob_mat, V,
        "... queries key_num, ... key_num d_v -> ... queries d_v"
    )
    return res

class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # weight initizalition
        std = 2.0 / (in_features + out_features)
        std_sqrt = math.sqrt(std)
        self.weight: Parameter = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((out_features, in_features), dtype=dtype, device=device),
                mean=0,
                std=std,
                a=-3 * std_sqrt,
                b=3 * std_sqrt,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  Apply the linear transformation to the input
        return x @ self.weight.T


class Embedding(Module):
    def __init__(
        self,
        num_embedding: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight: Parameter = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((num_embedding, embedding_dim), dtype=dtype, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # ... -> ..., embedding_dim
        return torch.index_select(self.weight, 0, token_ids.flatten()).view(
            (*token_ids.shape, -1)
        )


class RMSNorm(Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight: Parameter = Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, d_model)
        in_type = x.dtype
        x.to(torch.float32)

        tg = einops.rearrange(self.weight, "d_model -> 1 1 d_model")
        mean_sum_sqare = torch.sqrt(torch.mean(x * x, -1, keepdim=True) + self.eps)
        return (x * tg / mean_sum_sqare).to(in_type)


class SwiGLU(Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_ff == None:
            d_ff = int(d_model * 8.0 / 3)
            if d_ff % 64 != 0:
                d_ff = (d_ff // 64 + 1) * 64

        self.w1 = Linear(
            in_features=d_model, out_features=d_ff, dtype=dtype, device=device
        )

        self.w2 = Linear(
            in_features=d_ff, out_features=d_model, dtype=dtype, device=device
        )

        self.w3 = Linear(
            in_features=d_model, out_features=d_ff, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, d_model)
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)
        x2 = self.w3(x)
        return self.w2(x1 * x2)


class RoPE(Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        # diag
        rope_mat = torch.empty(max_seq_len, d_k // 2, 2, 2, device=device)
        for i in range(max_seq_len):
            for k in range(1, d_k // 2 + 1):
                theta_k = i / theta ** ((2 * k - 2) / d_k)
                rope_mat[i][k - 1] = torch.tensor(
                    [
                        [math.cos(theta_k), -math.sin(theta_k)],
                        [math.sin(theta_k), math.cos(theta_k)],
                    ],
                    device=device,
                )
        self.register_buffer("rope_mat", rope_mat)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions -> ... seq 
        rope_mat = self.get_buffer("rope_mat")[token_positions]
        x = einops.rearrange(x, "... seq_len (d_k_2 bs) -> ... seq_len d_k_2 bs", bs=2)
        y = einops.einsum(
            x,
            rope_mat,
            "... seq_len d_k_2 bs, ... seq_len d_k_2 rt bs -> ... seq_len d_k_2 rt",
        )
        return einops.rearrange(y, "... seq_len d_k_2 bs -> ... seq_len (d_k_2 bs)")

class MultiHeadSelfAttention(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_module: Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(
            in_features=d_model, out_features=self.d_k * self.num_heads, dtype=dtype, device=device)
        self.k_proj = Linear(
            in_features=d_model, out_features=self.d_k * self.num_heads, dtype=dtype, device=device)
        self.v_proj = Linear(
            in_features=d_model, out_features=self.d_v * self.num_heads, dtype=dtype, device=device)
        self.output_proj = Linear(
            in_features=self.d_v * self.num_heads, out_features=d_model, dtype=dtype, device=device)
        self.rope = rope_module

    
    def forward(self, x: torch.Tensor, token_positions:torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q = einops.rearrange(
            self.q_proj(x), "... seq (h dk) -> ... h seq dk", h = self.num_heads
        )
        K = einops.rearrange(
            self.k_proj(x), "... seq (h dk) -> ... h seq dk", h = self.num_heads
        )
        prev_batches_before_seq = len(Q.shape) - 2
        if self.rope != None:
            if token_positions == None:
                token_positions = torch.arange(0, seq_len).view(*([1] * prev_batches_before_seq), seq_len)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        V = einops.rearrange(
            self.v_proj(x), "... seq (h dv) -> ... h seq dv", h = self.num_heads
        )
        mask = torch.tril(torch.ones(Q.shape[:-2] + (seq_len, seq_len), dtype=torch.bool))
        out = einops.rearrange(
            scaled_dot_product_attention(Q, K, V, mask),
            "... h seq dv -> ... seq (h dv)"
        )
        return self.output_proj(out)
 

class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        rope_module: Module | None = None,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_module=rope_module,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(
            d_model= d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        self.ln1 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype
        )
    
    def forward(self, x: torch.Tensor, token_positions:torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))

class Transformer(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ): 
        super().__init__()
        self.token_embeddings = Embedding(num_embedding=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        rope = RoPE(theta=rope_theta, d_k= d_model // num_heads, max_seq_len=context_length)
        self.layers = ModuleList(
            [TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_module=rope,
                device = None,
                dtype = None
            ) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, 
                 lr=1e-3, 
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8):
        defaults = {
            "lr": lr, 
            "beta1": betas[0], 
            "beta2": betas[1], 
            "weight_decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure is not None else 0
        for pg in self.param_groups:
            lr = pg["lr"] 
            beta1 = pg["beta1"] 
            beta2 = pg["beta2"] 
            weight_decay = pg["weight_decay"]
            eps = pg["eps"]
            for p in pg["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # iteration number
                t = state.get("t", 1)
                grad = p.grad.data
                m = state.get("m", torch.zeros(p.data.shape, requires_grad=False, device=p.data.device, dtype=p.data.dtype))
                v = state.get("v", torch.zeros(p.data.shape, requires_grad=False, device=p.data.device, dtype=p.data.dtype))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v


def cosine_annealing(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        if it < warmup_iters:
            return max_learning_rate * it / warmup_iters
        elif it <= cosine_cycle_iters:
            t = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
            return min_learning_rate + 0.5 * (1 + math.cos(t)) * (max_learning_rate - min_learning_rate)
        else:
            return min_learning_rate

def gradient_cilpping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    square_sum = 0
    for p in parameters:
        if p.grad != None:
            # l2_norm = torch.sqrt(p.grad * p.grad)
            # scale_down = torch.ones(p.grad.shape) * max_l2_norm / (l2_norm + eps)
            # scale_down[l2_norm < max_l2_norm] = 1
            # p.grad = p.grad * scale_down
            square_sum += torch.sum(p.grad * p.grad)
    l2_norm = math.sqrt(square_sum)
    if square_sum > max_l2_norm:
        for p in parameters:
            if p.grad != None:
                p.grad = p.grad * max_l2_norm / (l2_norm + eps)

def test_linear():
    linear = Linear(in_features=10, out_features=20)
    empty_tensor = torch.empty((20, 10))
    linear.load_state_dict({"weight": empty_tensor})
    assert torch.all(linear.weight == empty_tensor)
    out = linear.forward(torch.rand(10, 10))
    assert out.shape == (10, 20), f"invalid {out.shape}"


def test_RMS():
    d_model = 10
    rms_norm = RMSNorm(d_model=d_model)
    target_shape = (2, 2, d_model)
    assert rms_norm.forward(torch.zeros(*target_shape)).shape == target_shape


def test_SwiGLU():
    d_model = 10
    swiglu = SwiGLU(d_model=d_model)
    state_dict = swiglu.state_dict()
    assert state_dict != None


def test_Rope():
    d_k = 10
    max_len = 10
    batch = 5
    seq_num = 5
    rope = RoPE(0.1, max_len, d_k)
    # b, s, d
    in_feat = torch.rand(batch, seq_num, d_k)
    tk_pos = torch.range(0, seq_num - 1, dtype=torch.int32)
    out = rope.forward(in_feat, tk_pos)
    assert out.shape == (batch, seq_num, d_k)

def test_sdpa():
    n = 10
    m = 5
    dk = 5
    dv = 5
    Q = torch.ones(n, dk)
    K = torch.ones(m, dk)
    V = torch.ones(m, dv)
    out = scaled_dot_product_attention(Q, K, V)
    assert out != None

def test_MultiHeadSelfAttention():
    max_len = 10
    num_heads = 4
    d_model = 16
    seq_len = 2
    batch = 1
    d_k = d_model //  num_heads
    rope = RoPE(
        0.1, 
        d_k=d_k,
        max_seq_len=max_len
    )
    mha = MultiHeadSelfAttention(d_model, num_heads, rope)
    mha.q_proj.weight.data = torch.eye(d_model)
    mha.k_proj.weight.data = torch.eye(d_model)
    mha.v_proj.weight.data = torch.eye(d_model)
    mha.output_proj.weight.data = torch.eye(d_model)

    token_positions = torch.arange(0, seq_len)
    input = torch.ones(batch, seq_len, d_model)
    out = mha.forward(input, None)
    assert out != None

def test_tb():
    batch = 1
    d_model = 16
    num_heads = 4
    max_len = 10
    seq_len = 2
    d_k = d_model //  num_heads
    rope = RoPE(
        0.1, 
        d_k=d_k,
        max_seq_len=max_len
    )
    input = torch.ones(batch, seq_len, d_model)
    tb = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        rope_module=rope
    )
    out = tb.forward(input)
    assert out != None

def test_sgd():
    from torch.optim.sgd import SGD
    for lr in [1, 10, 100]:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        print(f'{"=" * 10 }run with learning {lr}')
        for t in range(100):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.

def test_gc():
    t = torch.nn.Parameter(torch.randn(5,5))
    loss_c = t.sum()
    loss_c.backward()
    gradient_cilpping([t], 0.2)
    

if __name__ == "__main__":
    # test_linear()
    # test_RMS()
    # test_SwiGLU()
    # test_Rope()
    # test_sdpa()
    # test_MultiHeadSelfAttention()
    # test_tb()
    # test_sgd()
    test_gc()


