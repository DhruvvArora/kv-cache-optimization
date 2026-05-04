import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

PROMPT_LENGTHS = [512, 768, 1024]
MAX_NEW_TOKENS = 64
RUNS_PER_SETTING = 3


def make_prompt(tokenizer, target_tokens: int) -> str:
    base_text = (
        "Large language models rely on key value caches during decoding. "
        "This project studies how prompt length affects decode latency, memory use, "
        "and overall inference efficiency. "
    )
    text = base_text
    while True:
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if len(tokens) >= target_tokens:
            break
        text += base_text

    trimmed_ids = tokens[:target_tokens]
    return tokenizer.decode(trimmed_ids, skip_special_tokens=True)


def run_experiment():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    print("Model device:", next(model.parameters()).device)
    model.eval()

    # Warm-up
    warmup_prompt = "This is a warmup run for KV cache benchmarking."
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    results = []

    for prompt_len in PROMPT_LENGTHS:
        prompt = make_prompt(tokenizer, prompt_len)

        for run_idx in range(RUNS_PER_SETTING):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            total_latency = end_time - start_time
            avg_decode_sec_per_token = total_latency / max(output_tokens, 1)

            peak_vram_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if DEVICE == "cuda" else 0.0
            )

            results.append({
                "prompt_tokens": input_tokens,
                "generated_tokens": output_tokens,
                "total_latency_sec": round(total_latency, 4),
                "avg_decode_sec_per_token": round(avg_decode_sec_per_token, 4),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "run": run_idx + 1
            })

            print(
                f"Prompt={input_tokens}, Run={run_idx + 1}, "
                f"Latency={total_latency:.4f}s, "
                f"Avg/token={avg_decode_sec_per_token:.4f}s, "
                f"Peak VRAM={peak_vram_mb:.2f} MB"
            )

    df = pd.DataFrame(results)
    df.to_csv("kv_three_contexts_results.csv", index=False)
    print("\nSaved results to kv_three_contexts_results.csv")

    summary = df.groupby("prompt_tokens", as_index=False).agg({
        "generated_tokens": "mean",
        "total_latency_sec": "mean",
        "avg_decode_sec_per_token": "mean",
        "peak_vram_mb": "mean"
    })
    summary.to_csv("kv_three_contexts_summary.csv", index=False)
    print("Saved summary to kv_three_contexts_summary.csv")
    print("\nSummary:")
    print(summary)

    # Plot 1: Prompt length vs total latency
    plt.figure(figsize=(8, 5))
    plt.plot(summary["prompt_tokens"], summary["total_latency_sec"], marker="o")
    plt.xlabel("Prompt Length (tokens)")
    plt.ylabel("Total Latency (seconds)")
    plt.title("Prompt Length vs Total Latency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("latency_vs_prompt_length.png")
    plt.close()

    # Plot 2: Prompt length vs peak VRAM
    plt.figure(figsize=(8, 5))
    plt.plot(summary["prompt_tokens"], summary["peak_vram_mb"], marker="o")
    plt.xlabel("Prompt Length (tokens)")
    plt.ylabel("Peak VRAM (MB)")
    plt.title("Prompt Length vs Peak VRAM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vram_vs_prompt_length.png")
    plt.close()

    print("Saved charts: latency_vs_prompt_length.png, vram_vs_prompt_length.png")


if __name__ == "__main__":
    run_experiment()