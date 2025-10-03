import torch
from torch.nn import Module, Parameter, ModuleList
import math
import einops
from jaxtyping import Bool, Float, Int
from torch import Tensor
from typing import Optional, Callable, Iterable, BinaryIO, IO
import numpy.typing as npt
import os
import numpy as np
from logging import Logger
import random
import json
import logging
from cs336_basics.tokenizer import Tokenizer

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    sum_x = torch.sum(x, dim=dim, keepdim=True)
    return x / sum_x


def cross_entropy_loss(
    logits: Float[Tensor, " batch_size vocab_size"], y: Float[Tensor, " batch_size "]
) -> torch.Tensor:
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    sum_logits = torch.sum(torch.exp(logits), dim=-1, keepdim=True)
    y = einops.rearrange(y, "... batch_size -> ... batch_size 1")
    y_logits = torch.gather(logits, 1, y)
    return torch.mean(torch.log(sum_logits) - y_logits)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    dot_product_mat = einops.einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    ) / math.sqrt(d_k)
    if mask != None:
        dot_product_mat[~mask] = -float("inf")
    prob_mat = softmax(dot_product_mat, dim=-1)
    res = einops.einsum(
        prob_mat, V, "... queries key_num, ... key_num d_v -> ... queries d_v"
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
        self.weight: Parameter = Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(
            in_features=d_model,
            out_features=self.d_k * self.num_heads,
            dtype=dtype,
            device=device,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=self.d_k * self.num_heads,
            dtype=dtype,
            device=device,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=self.d_v * self.num_heads,
            dtype=dtype,
            device=device,
        )
        self.output_proj = Linear(
            in_features=self.d_v * self.num_heads,
            out_features=d_model,
            dtype=dtype,
            device=device,
        )
        self.rope = rope_module

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q = einops.rearrange(
            self.q_proj(x), "... seq (h dk) -> ... h seq dk", h=self.num_heads
        )
        K = einops.rearrange(
            self.k_proj(x), "... seq (h dk) -> ... h seq dk", h=self.num_heads
        )
        prev_batches_before_seq = len(Q.shape) - 2
        if self.rope != None:
            if token_positions == None:
                token_positions = torch.arange(0, seq_len).view(
                    *([1] * prev_batches_before_seq), seq_len
                )
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        V = einops.rearrange(
            self.v_proj(x), "... seq (h dv) -> ... h seq dv", h=self.num_heads
        )
        mask = torch.tril(
            torch.ones(Q.shape[:-2] + (seq_len, seq_len), dtype=torch.bool)
        )
        out = einops.rearrange(
            scaled_dot_product_attention(Q, K, V, mask),
            "... h seq dv -> ... seq (h dv)",
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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_module=rope_module,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            num_embedding=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        rope = RoPE(
            theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length
        )
        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope_module=rope,
                    device=None,
                    dtype=None,
                )
                for _ in range(num_layers)
            ]
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
    def __init__(
        self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    ):
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps,
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
                m = state.get(
                    "m",
                    torch.zeros(
                        p.data.shape,
                        requires_grad=False,
                        device=p.data.device,
                        dtype=p.data.dtype,
                    ),
                )
                v = state.get(
                    "v",
                    torch.zeros(
                        p.data.shape,
                        requires_grad=False,
                        device=p.data.device,
                        dtype=p.data.dtype,
                    ),
                )
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


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
        return min_learning_rate + 0.5 * (1 + math.cos(t)) * (
            max_learning_rate - min_learning_rate
        )
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


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str = "cpu"
):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    import random

    # (0 -> len - context_length - 1)
    possible_starting_indices = len(dataset) - context_length - 1
    # sample starting index
    start_index_list = random.sample(range(possible_starting_indices + 1), batch_size)
    xs = []
    ys = []
    for start in start_index_list:
        xs.extend(dataset[start : start + context_length])
        ys.extend(dataset[start + 1 : start + 1 + context_length])
        assert start + 1 + context_length <= len(dataset)

    return (
        torch.tensor(xs, device=device).view(batch_size, context_length),
        torch.tensor(ys, device=device).view(batch_size, context_length),
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_params = model.state_dict()
    opitmizer_state = optimizer.state_dict()
    torch.save(
        {
            "model_params": model_params,
            "opitmizer_state": opitmizer_state,
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model_params"])
    optimizer.load_state_dict(obj["opitmizer_state"])
    return obj["iteration"]


class Trainer:
    def __init__(
        self,
        # model config
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        # train loop config
        train_dataset_path: str,
        validate_dataset_path: str,
        max_iter: int,
        device: str,
        batch_size: int,
        logger: Logger,
        # optimizer config
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        max_l2_norm: float | None = None,
        # lr scheduler
        lr_scheduler_config: dict | None = None,
        # ckpt config
        ckpt_out_dir: str | None = None,
        resume_ckpt_path: str | None = None,
    ) -> None:
        self.device = device
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.model = Transformer(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=torch.device(self.device),
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.max_l2_norm = max_l2_norm
        self.lr_scheduler_config = lr_scheduler_config
        self.optmizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        self.train_dataset_path = train_dataset_path
        self.validate_dataset_path = validate_dataset_path
        self.train_dataset = np.load(file=train_dataset_path, mmap_mode="r")
        self.validate_dataset = np.load(file=validate_dataset_path, mmap_mode="r")
        self.max_iter = max_iter
        self.start_iter = 0
        self.ckpt_out_dir = ckpt_out_dir
        self.batch_size = batch_size
        self.logger = logger
        self.ckpt_interval = 5
        if resume_ckpt_path != None:
            with open(resume_ckpt_path, "rb") as f:
                self.start_iter = load_checkpoint(f, self.model, self.optmizer)

    @classmethod
    def from_config(cls, config: dict, logger: Logger | None = None) -> "Trainer":
        if logger == None:
            logger = logging.getLogger()
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler()],
            )
        return Trainer(logger=logger, **config)

    def dump_config(self):
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
            # train loop config
            "train_dataset_path": self.train_dataset_path,
            "validate_dataset_path": self.validate_dataset_path,
            "max_iter": self.max_iter,
            "device": self.device,
            "batch_size": self.batch_size,
            # optimizer config
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "max_l2_norm": self.max_l2_norm,
            # lr scheduler
            "lr_scheduler_config": self.lr_scheduler_config,
            # ckpt config
            "ckpt_out_dir": self.ckpt_out_dir,
        }

    def validate(self):
        validate_batches_num = len(self.validate_dataset) // 10
        with torch.no_grad():
            total_loss = 0
            for i in range(validate_batches_num):
                self.model.eval()
                x, y = get_batch(
                    dataset=self.validate_dataset,
                    batch_size=self.batch_size,
                    context_length=self.context_length,
                    device=self.device,
                )
                out = self.model(x)
                loss = cross_entropy_loss(
                    einops.rearrange(
                        out, "... batch seq vocab -> ... (batch seq) vocab"
                    ),
                    einops.rearrange(y, "... batch seq -> ... (batch seq)"),
                )
                self.logger.info(
                    f"[Validate :{i + 1} / {validate_batches_num}]: {{loss : {loss}, batch_num: {self.batch_size}}}"
                )
                total_loss += loss
            return total_loss / validate_batches_num
        pass

    def train(self):
        self.logger.info(
            f"Start training from iteration : {self.start_iter + 1} "
            + f"with config : \n ${json.dumps(self.dump_config(), indent=2)}"
        )
        cur_lr = self.lr
        for cur_iter in range(self.start_iter + 1, self.max_iter + 1):
            LOG_PREFIX = f"[{cur_iter} / {self.max_iter}]:"
            self.logger.info(f"{LOG_PREFIX} Getting batch with size {self.batch_size}")
            x, y = get_batch(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                context_length=self.context_length,
                device=self.device,
            )
            # batch seq vocab
            self.logger.info(f"{LOG_PREFIX} Model Fowarding ...")
            self.model.train()
            out = self.model(x)
            loss = cross_entropy_loss(
                einops.rearrange(out, "... batch seq vocab -> ... (batch seq) vocab"),
                einops.rearrange(y, "... batch seq -> ... (batch seq)"),
            )
            self.logger.info(f"{LOG_PREFIX} Model Backwarding ...")
            self.optmizer.zero_grad()
            loss.backward()
            if self.max_l2_norm != None:
                gradient_cilpping(self.model.parameters(), self.max_l2_norm)
            self.logger.info(f"{LOG_PREFIX} Optimizer Updating ...")
            self.optmizer.step()
            if self.lr_scheduler_config != None:
                if self.lr_scheduler_config["name"] == "cosine":
                    cur_lr = cosine_annealing(
                        cur_iter,
                        max_learning_rate=self.lr_scheduler_config["max_learning_rate"],
                        min_learning_rate=self.lr_scheduler_config["min_learning_rate"],
                        warmup_iters=self.lr_scheduler_config["warmup_iters"],
                        cosine_cycle_iters=self.lr_scheduler_config[
                            "cosine_cycle_iters"
                        ],
                    )
                    for pg in self.optmizer.param_groups:
                        pg["lr"] = cur_lr
                else:
                    raise ValueError("Invalid lr schduler name")
            self.logger.info(f"{LOG_PREFIX} {{loss : {loss}, lr: {cur_lr}}}")
            if self.ckpt_out_dir != None and cur_iter % self.ckpt_interval == 0:
                validate_loss = self.validate()
                self.logger.info(
                    f"{LOG_PREFIX} {{validate_loss : {validate_loss}, iter: {cur_iter}}}"
                )
                out_ckpt_path = os.path.join(self.ckpt_out_dir, f"ckpt-{cur_iter}.dump")
                with open(out_ckpt_path, "wb") as f:
                    self.logger.info(
                        f"{LOG_PREFIX} Saving ckpt into {out_ckpt_path} ..."
                    )
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optmizer,
                        iteration=cur_iter,
                        out=f,
                    )
        # save final model
        if self.ckpt_out_dir != None:
            final_model_path = os.path.join(self.ckpt_out_dir, "ckpt-final.dump")
            with open(final_model_path, "wb") as f:
                self.logger.info(f"Saving final model into {final_model_path} ...")
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optmizer,
                    iteration=self.max_iter,
                    out=f,
                )


class Decoder:
    def __init__(
        self, model: Module, tokenizer: Tokenizer, device: str = "cpu"
    ) -> None:
        self.model = model
        self.tokenizer: Tokenizer = tokenizer
        self.device = device
        self.end_token = 1

    def decode(self, prompt: str, max_output_len: int, tempeature: float, top_p: int):
        input_tensor = einops.rearrange(
            torch.tensor(
                self.tokenizer.encode(prompt),
                dtype=torch.int64,
                device=torch.device(self.device),
            ),
            "seq -> 1 seq",
        )
        prompt_token_len = input_tensor.shape[-1]
        for _ in range(max_output_len):
            # batch, seq, vocab
            out = self.model(input_tensor)
            # [vocab,]
            logits = out[0][-1] / tempeature
            prob = softmax(logits, dim=-1)
            prob_list = prob.flatten().tolist()
            top_p_val = sorted(prob_list)[-top_p]
            prob[prob < top_p_val] = 0
            prob = prob / torch.sum(prob)
            next_token = torch.multinomial(prob, num_samples=1)
            if next_token[0].detach().cpu() == self.end_token:
                break
            input_tensor = torch.cat(
                [input_tensor, einops.rearrange(next_token, "... seq -> ... 1 seq")],
                dim=1,
            )
        return self.tokenizer.decode(
            input_tensor[0][prompt_token_len:].cpu().detach().tolist()
        )


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
    d_k = d_model // num_heads
    rope = RoPE(0.1, d_k=d_k, max_seq_len=max_len)
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
    d_k = d_model // num_heads
    rope = RoPE(0.1, d_k=d_k, max_seq_len=max_len)
    input = torch.ones(batch, seq_len, d_model)
    tb = TransformerBlock(d_model=d_model, num_heads=num_heads, rope_module=rope)
    out = tb.forward(input)
    assert out != None


def test_sgd():
    from torch.optim.sgd import SGD

    for lr in [1, 10, 100]:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        print(f'{"=" * 10 }run with learning {lr}')
        for t in range(100):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.


def test_gc():
    t = torch.nn.Parameter(torch.randn(5, 5))
    loss_c = t.sum()
    loss_c.backward()
    gradient_cilpping([t], 0.2)


def test_training_loop():
    output_dir = "./data/train_debug"
    dataset_path = os.path.join(output_dir, "tiny_debug_dataset.npy")
    if os.path.exists(dataset_path):
        # remove existing numpy file
        os.remove(dataset_path)
    with open(dataset_path, "wb") as f:
        test_arr = np.arange(0, 100, dtype=np.int64)
        np.save(f, test_arr)
    train_config = {
        "vocab_size": 100,
        "context_length": 20,
        "d_model": 20,
        "num_layers": 2,
        "num_heads": 2,
        "d_ff": 30,
        "rope_theta": 0.1,
        # train loop config
        "dataset_path": dataset_path,
        "max_iter": 10,
        "device": "cpu",
        "batch_size": 32,
        # optimizer config
        "lr": 1e-3,
        "weight_decay": 0.1,
        "betas": (0.1, 0.2),
        "eps": 1e-6,
        "max_l2_norm": None,
        # lr scheduler
        "lr_scheduler_config": {
            "name": "cosine",
            "max_learning_rate": 1e-2,
            "min_learning_rate": 1e-3,
            "warmup_iters": 10,
            "cosine_cycle_iters": 60,
        },
        # ckpt config
        "ckpt_out_dir": output_dir,
    }
    trainer = Trainer.from_config(train_config)
    trainer.train()


def test_decoder():
    import pickle

    tokenizer_result_dir = "/home/wuziyi/code/cs336/assignment1-basics/data/vocab_result/20250928_184448_180765"
    vocab_dump_path = os.path.join(tokenizer_result_dir, "vocab.pkl")
    merges_dump_path = os.path.join(tokenizer_result_dir, "merges.pkl")

    with open(vocab_dump_path, "rb") as fv, open(merges_dump_path, "rb") as fm:
        vocab = pickle.load(fv)
        merges = pickle.load(fm)
        tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
        vocab_size = len(vocab)
        context_length = 100
        d_model = 128
        num_layers = 3
        num_heads = 2
        d_ff = 152
        rope_theta = 1e3
        model = Transformer(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=torch.device("cpu"),
        )
        model.eval()
        test_decoder = Decoder(model, tokenizer=tokenizer)
        print(test_decoder.decode("Hello", 10, 10, 2))


if __name__ == "__main__":
    # test_linear()
    # test_RMS()
    # test_SwiGLU()
    # test_Rope()
    # test_sdpa()
    # test_MultiHeadSelfAttention()
    # test_tb()
    # test_sgd()
    # test_gc()
    # test_training_loop()
    test_decoder()
