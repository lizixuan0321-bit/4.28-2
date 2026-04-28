import argparse
import time

import torch

from models.msfn import MSFN


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def format_num(num: float) -> str:
    if num >= 1e9:
        return f"{num / 1e9:.4f} G"
    if num >= 1e6:
        return f"{num / 1e6:.4f} M"
    if num >= 1e3:
        return f"{num / 1e3:.4f} K"
    return f"{num:.0f}"


@torch.no_grad()
def profile_latency_and_memory(
    module: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup: int,
    repeats: int,
    use_cuda: bool,
):
    module.eval()

    for _ in range(warmup):
        _ = module(input_tensor)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(repeats):
        _ = module(input_tensor)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / repeats) * 1000.0
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if use_cuda else float("nan")
    )
    return avg_ms, peak_mem_mb


def profile_flops(module: torch.nn.Module, input_tensor: torch.Tensor):
    try:
        from thop import profile  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'thop'. Install with: pip install thop"
        ) from exc
    macs, _ = profile(module, inputs=(input_tensor,), verbose=False)
    flops = macs * 2.0
    return flops


def build_parser():
    parser = argparse.ArgumentParser(
        description="Profile lightweight FSN (DSConv) efficiency only."
    )
    parser.add_argument("--num_token", type=int, default=4, help="Tokenizer number n (paper default: 4).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Profiling device.")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations.")
    parser.add_argument("--repeats", type=int, default=200, help="Measured iterations.")
    return parser


def main():
    args = build_parser().parse_args()

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # FSN in this project is used at two scales:
    # msfn1: C=320, HxW=10x10 (global) and 5x5 (local patch)
    # msfn2: C=640, HxW=5x5 (global) and 3x3 (local patch)
    cases = [
        ("msfn1_global", MSFN(320, args.num_token), (1, 320, 10, 10)),
        ("msfn1_local", MSFN(320, args.num_token), (1, 320, 5, 5)),
        ("msfn2_global", MSFN(640, args.num_token), (1, 640, 5, 5)),
        ("msfn2_local", MSFN(640, args.num_token), (1, 640, 3, 3)),
    ]

    print("=== Lightweight FSN Efficiency Report (DSConv version) ===")
    print(f"device={device.type}  num_token={args.num_token}  warmup={args.warmup}  repeats={args.repeats}")
    print("")

    total_params = 0
    total_flops = 0.0
    total_avg_ms = 0.0
    peak_mems = []

    for name, module, shape in cases:
        module = module.to(device)
        x = torch.randn(*shape, device=device)

        params = count_params(module)
        flops = profile_flops(module, x)
        avg_ms, peak_mem_mb = profile_latency_and_memory(
            module, x, warmup=args.warmup, repeats=args.repeats, use_cuda=use_cuda
        )

        total_params += params
        total_flops += flops
        total_avg_ms += avg_ms
        if use_cuda:
            peak_mems.append(peak_mem_mb)

        print(
            f"[{name}] input={shape} | Params={format_num(params)} | "
            f"FLOPs={format_num(flops)} | Time={avg_ms:.4f} ms"
            + (f" | PeakMem={peak_mem_mb:.2f} MB" if use_cuda else "")
        )

    print("")
    print("=== Aggregated (sum over 4 FSN calls) ===")
    print(f"Params: {format_num(total_params)}")
    print(f"FLOPs : {format_num(total_flops)}")
    print(f"Time  : {total_avg_ms:.4f} ms")
    if use_cuda and peak_mems:
        print(f"PeakMem(max over cases): {max(peak_mems):.2f} MB")


if __name__ == "__main__":
    main()
