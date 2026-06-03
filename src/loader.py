"""HF loaders for TinyLlama + the WikiText perplexity dataset."""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_tokenizer(model_id=MODEL_ID):
    tok = AutoTokenizer.from_pretrained(model_id)
    # Llama tokenizers ship without a pad token. Reusing eos shuts up
    # the warnings during generate().
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_fp16_model(model_id=MODEL_ID):
    """FP16 baseline. Everything else is compared against this."""
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def load_int8_model(model_id=MODEL_ID):
    cfg = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg, device_map="auto"
    )


def load_int4_model(model_id=MODEL_ID, quant_type="nf4"):
    # NF4 + double quant is the QLoRA recipe. compute_dtype=fp16 keeps
    # the matmul in half precision and only the storage is INT4.
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg, device_map="auto"
    )


def load_wikitext_test_text():
    """WikiText-2 test split, concatenated. Standard ppl benchmark."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def load_wikitext_calibration_texts(n=128, min_length=200):
    # Train split, separate from the test set we use for ppl.
    # Filter out the section-header one-liners that WikiText is full of.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    long_enough = [t for t in ds["text"] if len(t) > min_length]
    return long_enough[:n]
