import torch
import time
import math
from cute_flash_attention_101 import mha_fwd
from torch.nn import functional as F

torch.manual_seed(42)

def create_tensors(b, h, s, d):
    return [torch.randn((b, h, s, d), dtype=torch.float16, device="cuda").requires_grad_() for _ in range(3)]

def naive_attention(q, k, v, sm_scale, is_causal):
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if is_causal:
        mask = torch.triu(torch.ones(q.size(-2), q.size(-2), device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

def time_function(func, *args):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = func(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - start, result

def run_test(b=2, h=16, s=1024, d=64, is_causal=True):
    q, k, v = create_tensors(b, h, s, d)
    sm_scale = 1.0 / math.sqrt(d)

    torch_time, torch_out = time_function(naive_attention, q, k, v, sm_scale, is_causal)
    flash_time, flash_out = time_function(mha_fwd, q, k, v, sm_scale, is_causal)

    print(f"Test: b={b}, h={h}, s={s}, d={d}")
    print(f"Naive Torch: {torch_time*1000:.2f}ms")
    print(f"CuTe Flash:  {flash_time*1000:.2f}ms")
    print(f"Speedup: {torch_time/flash_time:.2f}x")
    print("-" * 80)

    assert torch.allclose(torch_out, flash_out[0], atol=1e-2), "Outputs don't match"

if __name__ == "__main__":
    for b in [1, 2, 4]:
        for h in [8, 16, 32]:
            for s in [512, 1024, 2048]:
                for d in [32, 64]:
                        run_test(b, h, s, d)