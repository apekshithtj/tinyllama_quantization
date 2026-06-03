"""perplexity, throughput, peak memory."""

import time

import torch


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, text=None, max_length=2048, n_chunks=None):
    """
    Sliding window ppl on a long string.

    n_chunks caps how many windows we evaluate (used during the sweep,
    where doing the full test set 22 times would take forever).
    Pass None for the full thing.
    """
    if text is None:
        from .loader import load_wikitext_test_text
        text = load_wikitext_test_text()

    enc = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    total = enc.size(1)

    nlls = []
    tokens_seen = 0
    done = 0

    for start in range(0, total, max_length):
        if n_chunks is not None and done >= n_chunks:
            break
        chunk = enc[:, start:start + max_length]
        if chunk.size(1) < 2:
            continue
        out = model(chunk, labels=chunk)
        # out.loss is the mean nll over this chunk; multiply by len so
        # the final average weights longer chunks correctly.
        nlls.append(out.loss * chunk.size(1))
        tokens_seen += chunk.size(1)
        done += 1

    if not nlls:
        raise ValueError("nothing to evaluate")

    return torch.exp(torch.stack(nlls).sum() / tokens_seen).item()


def measure_throughput(model, tokenizer, prompt="The capital of France is",
                       new_tokens=128, n_warmup=3):
    """tokens/sec, greedy. Warmup matters: skipping it broke my first runs."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    for _ in range(n_warmup):
        _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    _ = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return new_tokens / (time.time() - t0)


def measure_peak_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def summarize_run(model, tokenizer, label, n_chunks=20):
    """Run all three metrics and return them as a dict."""
    reset_peak_memory()
    ppl = evaluate_perplexity(model, tokenizer, n_chunks=n_chunks)
    tps = measure_throughput(model, tokenizer)
    peak = measure_peak_memory_gb()
    row = {"label": label, "perplexity": ppl, "tokens_per_sec": tps, "peak_memory_gb": peak}
    print(f"[{label}] ppl={ppl:.3f}  tok/s={tps:.2f}  peak={peak:.2f} GB")
    return row
