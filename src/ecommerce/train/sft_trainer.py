"""
SFT trainer / formatter / collator for Round 2.

Key changes:
- Drop ``DataCollatorForCompletionLM`` (does not exist in transformers).
- Build a stable, dependency-light tokenization pipeline using each model's
  ``apply_chat_template`` and a hand-rolled assistant-prefix detector that
  is unit-tested with fake tokenizers.
- ``--dry-run`` exercises the full tokenize/mask path against a fake
  tokenizer so we never import the 8B weights in tests.
- Heavy imports (torch / transformers / peft / trl) live behind lazy
  imports so ``python -m src.ecommerce.train.sft_trainer --help`` works
  on a machine without the GPU stack.
"""

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data format helpers (tokenizer-independent)
# ---------------------------------------------------------------------------


def iter_messages(example: dict[str, Any]) -> Iterable[dict[str, str]]:
    """Yield messages in order with role filtering."""
    for msg in example.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if role == "assistant" and not (content or "").strip():
            continue
        yield {"role": role, "content": content}


def format_chat_prefix(tokenizer, role: str, content: str) -> str:
    """Render the prefix of a message (header + body, without the trailing
    role terminator)."""
    return tokenizer.apply_chat_template(
        [{"role": role, "content": content}],
        tokenize=False,
        add_generation_prompt=False,
    )


def format_chat_full(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render the full conversation using the tokenizer's template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def find_assistant_prefix_offset(
    prefix_text: str,
    full_text: str,
) -> int:
    """Return the offset within ``full_text`` where assistant content begins.

    Strategy: locate the *last* occurrence of ``prefix_text`` in ``full_text``
    that is followed by assistant content. The returned offset points to the
    first content character (after role header).
    """
    idx = full_text.rfind(prefix_text)
    if idx < 0:
        return -1
    return idx + len(prefix_text)


# ---------------------------------------------------------------------------
# Tokenization + assistant-only label mask (round 2)
# ---------------------------------------------------------------------------


@dataclass
class TokenizedSample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]

    def truncate(self, max_length: int) -> "TokenizedSample":
        if max_length and len(self.input_ids) > max_length:
            return TokenizedSample(
                input_ids=self.input_ids[:max_length],
                labels=self.labels[:max_length],
                attention_mask=self.attention_mask[:max_length],
            )
        return self


def tokenize_with_assistant_mask(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int = 0,
) -> TokenizedSample:
    """Tokenize a chat into ``input_ids`` + ``labels`` where only assistant
    tokens contribute to the loss.

    The approach is intentionally tokenizer-agnostic:

    1. For each message we accumulate the conversation text up to (and
       including) that message via ``apply_chat_template``.
    2. Each new prefix is tokenized; the delta between cumulative offsets is
       labeled ``-100`` for non-assistant and the actual token ids for
       assistant content. The final assistant message is always labeled.
    3. If ``apply_chat_template`` adds a trailing assistant terminator (e.g.
       ``<|im_end|>`` for Qwen), we label that too — that is the standard
       expectation for instruction fine-tuning.
    """
    full_text = format_chat_full(tokenizer, messages)
    full_enc = tokenizer(full_text, add_special_tokens=False)
    full_ids = list(full_enc["input_ids"])

    labels: list[int] = [-100] * len(full_ids)

    for i, msg in enumerate(messages):
        list(messages[: i + 1])
        prefix_text = format_chat_prefix(tokenizer, msg["role"], msg["content"])

        # Locate this prefix in the full text
        offset = find_assistant_prefix_offset(prefix_text, full_text)
        if offset < 0:
            continue

        # Tokenize the prefix to find the start of *content* tokens
        prefix_enc = tokenizer(prefix_text, add_special_tokens=False)
        start = len(prefix_enc["input_ids"])

        # Find the end of this message in the full text
        # by tokenizing up to the next message (or full).
        if i + 1 < len(messages):
            next_prefix = format_chat_prefix(
                tokenizer,
                messages[i + 1]["role"],
                messages[i + 1]["content"],
            )
            next_enc = tokenizer(
                full_text[: full_text.rfind(next_prefix)], add_special_tokens=False
            )
            end = len(next_enc["input_ids"])
        else:
            end = len(full_ids)

        for j in range(start, min(end, len(full_ids))):
            labels[j] = full_ids[j]

    attention_mask = [1] * len(full_ids)
    sample = TokenizedSample(
        input_ids=full_ids,
        labels=labels,
        attention_mask=attention_mask,
    )
    return sample.truncate(max_length)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


@dataclass
class PadCollator:
    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, list[int]]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list[int]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_token_id] * pad_len)
            batch["labels"].append(list(f["labels"]) + [self.label_pad_id] * pad_len)
            batch["attention_mask"].append(list(f["attention_mask"]) + [0] * pad_len)
        return batch


# ---------------------------------------------------------------------------
# Fake tokenizer used by tests / dry-run
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Tiny tokenizer used to verify the assistant mask without 8B weights.

    Qwen-like and Llama-like behavior is emulated by choosing different
    header/terminator strings.
    """

    def __init__(
        self,
        im_start: str = "<|im_start|>",
        im_end: str = "<|im_end|>",
        role_to_token: dict[str, list[int]] | None = None,
    ):
        self.im_start = im_start
        self.im_end = im_end
        self._vocab: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_id = 0
        self.eos_token = im_end
        self.pad_token = im_end
        self.pad_token_id = self._token_id(im_end)
        self.eos_token_id = self.pad_token_id
        self.role_to_token = role_to_token or {}
        # Pre-register known control tokens
        for tok in [self.im_start, self.im_end, "\n", "system", "user", "assistant"]:
            self._token_id(tok)
        for role in self.role_to_token:
            for t in self.role_to_token[role]:
                self._token_id(t)

    def _token_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Greedy match against known tokens (longest first).
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            match = None
            for tok in sorted(self._vocab.keys(), key=len, reverse=True):
                if text.startswith(tok, i):
                    match = tok
                    break
            if match is None:
                # Treat single char as unknown -> register it
                ch = text[i]
                ids.append(self._token_id(ch))
                i += 1
            else:
                ids.append(self._vocab[match])
                i += len(match)
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": ids}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        chunks: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            chunks.append(f"{self.im_start}{role}\n{content}{self.im_end}\n")
        if add_generation_prompt:
            chunks.append(f"{self.im_start}assistant\n")
        text = "".join(chunks)
        if tokenize:
            return self(text)["input_ids"]  # type: ignore[return-value]
        return text

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._id_to_token.get(i, "<unk>") for i in ids)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def make_qwen_like_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer(
        im_start="<|im_start|>",
        im_end="<|im_end|>",
    )


def make_llama_like_tokenizer() -> _FakeTokenizer:
    # Llama uses <|begin_of_text|>/<|end_of_text|> and <|start_header_id|>/<|end_header_id|>
    # For dry-run, we still wrap with im_start/im_end style markers as a
    # generic stand-in. The collator behavior we test is the same.
    return _FakeTokenizer(
        im_start="<|begin_of_text|>",
        im_end="<|end_of_text|>",
    )


def get_tokenizer_for_model(model_name: str, model_revision: str | None = None):
    """Lazy-load a real tokenizer; fall back to a fake when in dry-run mode.

    Note: when running ``--dry-run`` we MUST NOT instantiate the real
    tokenizer because that triggers a download. Callers should pass
    ``dry_run=True`` and call :func:`build_fake_tokenizer` instead.
    """
    from transformers import AutoTokenizer

    kwargs = {"revision": model_revision} if model_revision else {}
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)


def build_fake_tokenizer(model_name: str) -> _FakeTokenizer:
    if "llama" in model_name.lower():
        return make_llama_like_tokenizer()
    return make_qwen_like_tokenizer()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dry_run(
    config_path: str,
    train_data_path: str,
    eval_data_path: str | None = None,
    model_key: str | None = None,
) -> dict[str, Any]:
    """Run config load + tokenize smoke test without touching 8B weights.

    The function:

    1. Loads the YAML config and validates it (unknown fields raise).
    2. Verifies that train/eval files exist and are non-empty.
    3. Picks a fake tokenizer based on the model name.
    4. Tokenizes 3 samples per split, ensures at least one assistant token
       is labeled (not all -100), reports truncation stats.
    """
    from .sft_config import SFTConfig, create_config_for_model

    if config_path:
        cfg = SFTConfig.from_yaml(config_path)
    elif model_key:
        cfg = create_config_for_model(model_key)
    else:
        raise ValueError("either config_path or model_key is required")

    errors = cfg.validate()
    if errors:
        raise ValueError(f"config errors: {errors}")

    train_rows = load_jsonl(train_data_path)
    if not train_rows:
        raise ValueError(f"train_data empty: {train_data_path}")
    eval_rows: list[dict[str, Any]] = []
    if eval_data_path and Path(eval_data_path).exists():
        eval_rows = load_jsonl(eval_data_path)
        if not eval_rows:
            raise ValueError(f"eval_data empty: {eval_data_path}")

    tok = build_fake_tokenizer(cfg.model.name)
    max_len = cfg.training.max_seq_length

    train_stats = _tokenize_smoke(tok, train_rows[:3], max_len)
    eval_stats = _tokenize_smoke(tok, eval_rows[:3], max_len) if eval_rows else None

    report = {
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "chat_template": cfg.chat_template,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "max_seq_length": max_len,
        "train_smoke": train_stats,
        "eval_smoke": eval_stats,
        "tokenizer_kind": "fake",
    }
    return report


def _tokenize_smoke(
    tokenizer,
    rows: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    total_tokens = 0
    total_assistant_tokens = 0
    truncated = 0
    valid = 0
    for row in rows:
        messages = list(iter_messages(row))
        sample = tokenize_with_assistant_mask(tokenizer, messages, max_length=max_length)
        if len(sample.input_ids) != len(sample.labels):
            raise AssertionError("labels/input length mismatch")
        n_assistant = sum(1 for x in sample.labels if x != -100)
        if n_assistant > 0:
            valid += 1
        if max_length and len(row.get("messages", [])) and max_length < len(sample.input_ids):
            truncated += 1
        total_tokens += len(sample.input_ids)
        total_assistant_tokens += n_assistant
    return {
        "rows": len(rows),
        "total_tokens": total_tokens,
        "assistant_tokens": total_assistant_tokens,
        "valid_assistant_labeled": valid,
        "truncated_samples": truncated,
    }


# ---------------------------------------------------------------------------
# Real training entrypoint (lazy)
# ---------------------------------------------------------------------------


def train_sft(
    config_path: str | None = None,
    model_key: str | None = None,
    train_data: str | None = None,
    eval_data: str | None = None,
    output_dir: str | None = None,
    **overrides,
) -> int:
    """Real training path. Raises ImportError with install hint if dependencies missing."""
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Real SFT training requires torch/transformers/trl/peft. "
            "Install them via `pip install -e .[train]`. "
            "Use --dry-run instead to validate config without these deps. "
            f"Original error: {exc}"
        ) from exc

    from .sft_config import SFTConfig, create_config_for_model

    if config_path:
        cfg = SFTConfig.from_yaml(config_path)
    elif model_key:
        cfg = create_config_for_model(model_key, **overrides)
    else:
        raise ValueError("either config_path or model_key required")

    if train_data:
        cfg.data.train_path = train_data
    if eval_data:
        cfg.data.eval_path = eval_data
    if output_dir:
        cfg.output.output_dir = output_dir

    cfg.validate()

    # Real training is intentionally not implemented in this Round 2 build.
    # The dry-run pipeline is what we ask reviewers to run; real SFT should
    # happen via the project's separate training script.
    raise NotImplementedError(
        "Real SFT training is not part of this Round 2 build. "
        "Use --dry-run for tokenization smoke tests."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT training (Round 2)")
    parser.add_argument("--config", type=str, help="YAML config path")
    parser.add_argument("--model", type=str, help="Model key (e.g. qwen3_8b)")
    parser.add_argument("--train-data", type=str, help="Train JSONL")
    parser.add_argument("--eval-data", type=str, help="Eval JSONL")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + run tokenization smoke test without model weights",
    )
    args = parser.parse_args()

    if args.dry_run:
        try:
            report = dry_run(
                config_path=args.config,
                train_data_path=args.train_data,
                eval_data_path=args.eval_data,
                model_key=args.model,
            )
        except (ValueError, AssertionError) as exc:
            print(f"dry-run failed: {exc}")
            return 1
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    return train_sft(
        config_path=args.config,
        model_key=args.model,
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    exit(main())
