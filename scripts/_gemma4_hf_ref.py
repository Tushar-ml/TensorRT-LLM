"""HF transformers reference for gemma4-31b (text-only) — to localize the
TRTLLM plain-decode correctness bug.  Runs the SAME raw prompts greedily and
prints the first generated token ids + decoded text, so we can compare against
the TRTLLM smoke output:
  - if HF's first token differs from TRT's -> prefill/weight-loading bug
  - if first tokens match but HF stays coherent while TRT loops -> decode bug
"""
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

# Official unquantized bf16 checkpoint — the SAME one the TRTLLM smoke loads
# (GEMMA4_BACKBONE default). /home/ubuntu/gemma4-31b/model is FP8-quantized, a
# different checkpoint — do not use it here.
MODEL = "google/gemma-4-31b-it"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto").eval()

    prompts = ["What is the capital of France?",
               "Explain gravity in one sentence."]
    for p in prompts:
        # Proper chat formatting via the tokenizer's template (adds BOS + turns).
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(model.device)
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=40, do_sample=False)
        gen = out[0][ids.shape[1]:]
        print(f"--- prompt: {p!r}")
        print("   first 8 gen ids:", gen[:8].tolist())
        print("   ->", repr(tok.decode(gen, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
