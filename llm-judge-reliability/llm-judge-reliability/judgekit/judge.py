"""The judge. We never let the model WRITE a verdict: we read the probabilities it assigns
to the next token (A / B / C, or 1..5), so there is nothing to parse and no sampling noise.

Sign convention used everywhere ("margin"): positive = favors model_a, in the ORIGINAL frame.
  margin = 0.5 * [ (log pA - log pB)  +  (log sw_pB - log sw_pA) ]
The second term comes from the run with the answers swapped (letter A there is the original
model_b), so subtracting in the other order puts it back in the original frame. Averaging
the two orders cancels most order effects. This sign is a convention we chose, not a fact.
"""
import numpy as np

from .prompts import build_prompt
from .rows import swap_row

EPS = 1e-6  # float16 can underflow to exactly 0 probability -> log(0) = -inf -> nan. Clip.
LETTERS = ["A", "B", "C"]


def lg(x):
    return np.log(np.clip(x, EPS, None))


class HFJudge:
    """Hugging Face causal LM, one forward pass per prompt, results cached by prompt text."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", dtype: str = "float16",
                 device_map: str = "cuda", load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kw["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        self.letter_ids = {l: self.tok.encode(l, add_special_tokens=False)[0] for l in LETTERS}
        self.digit_ids = {d: self.tok.encode(str(d), add_special_tokens=False)[0]
                          for d in range(1, 6)}
        self._cache = {}

    def _next_token_logits(self, prompt: str):
        chat = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        inputs = self.tok(chat, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            return self.model(**inputs).logits[0, -1]

    def probs(self, prompt: str) -> dict:
        """{'A':p, 'B':p, 'C':p}, softmax over just those three tokens."""
        if prompt not in self._cache:
            logits = self._next_token_logits(prompt)
            three = self.torch.stack([logits[self.letter_ids[l]] for l in LETTERS])
            p = self.torch.softmax(three, dim=0).float().cpu().tolist()
            self._cache[prompt] = dict(zip(LETTERS, p))
        return self._cache[prompt]

    def expected_score(self, prompt: str) -> float:
        """Pointwise rubric: expected value of the digit 1..5 under the model's probabilities."""
        key = ("score", prompt)
        if key not in self._cache:
            logits = self._next_token_logits(prompt)
            five = self.torch.stack([logits[self.digit_ids[d]] for d in range(1, 6)])
            p = self.torch.softmax(five, dim=0).float().cpu().numpy()
            self._cache[key] = float((p * np.arange(1, 6)).sum())
        return self._cache[key]


def margin_from_probs(p: dict, sp: dict) -> float:
    return 0.5 * ((lg(p["A"]) - lg(p["B"])) + (lg(sp["B"]) - lg(sp["A"])))


def order_margins(judge, row, prompt_fn=build_prompt):
    """(margin in original order, margin in swapped order translated back). >0 favors model_a."""
    p = judge.probs(prompt_fn(row))
    sp = judge.probs(prompt_fn(swap_row(row)))
    return lg(p["A"]) - lg(p["B"]), lg(sp["B"]) - lg(sp["A"])


def margin(judge, row, prompt_fn=build_prompt) -> float:
    mo, ms = order_margins(judge, row, prompt_fn)
    return 0.5 * (mo + ms)
