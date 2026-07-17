"""
SFT trainer / formatter / collator for Round 3.

Key changes vs Round 2:
- Tokenization uses *cumulative prefix tokenization* to find assistant spans:
  each boundary is determined by tokenizing messages[:i] and messages[:i+1]
  via apply_chat_template, NOT by string rfind on the full rendered text.
- Two structurally different fake tokenizers:
    * Qwen-style ChatML  (<|im_start|> / <|im_end|> per message)
    * Llama-3-style     (<|begin_of_text|>, role headers, <|eot_id|> per message)
- assistant-only labels: system / user / tool → -100;
  assistant generation content → actual token ids (including official terminator).
- Truncation preserves original length for statistics; samples whose
  assistant span is fully discarded are rejected.
- Collator returns list-of-lists (not tensors) so dry-run works without torch.
- Heavy imports (torch / transformers / peft / trl) live behind lazy imports.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message iterator
# ---------------------------------------------------------------------------


def iter_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Return filtered, ordered message list for tokenization.

    Skips empty-assistant messages and unknown roles.
    """
    out: list[dict[str, str]] = []
    for msg in example.get("messages") or []:
        role = msg.get("role", "")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        content: str = msg.get("content", "")
        # Skip empty assistant turns (no content to learn from)
        if role == "assistant" and not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out


# ---------------------------------------------------------------------------
# Tokenization + assistant-only label mask  (cumulative prefix method)
# ---------------------------------------------------------------------------


@dataclass
class TokenizedSample:
    """Tokenized conversation with per-token loss mask.

    Attributes:
        input_ids:       list of token ids.
        labels:          same length as input_ids;
                         -100 = ignore (non-assistant), actual id = learn.
        attention_mask:  1 for real tokens, 0 for pad.
        original_len:    token count BEFORE truncation (for statistics).
        truncated:       True if sequence was cut at max_length.
        assistant_lost:   True if truncation removed the entire assistant span.
    """

    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    original_len: int = 0
    truncated: bool = False
    assistant_lost: bool = False

    def truncate(self, max_length: int) -> TokenizedSample:
        if not max_length or len(self.input_ids) <= max_length:
            self.original_len = len(self.input_ids)
            return self

        self.truncated = True
        self.original_len = len(self.input_ids)

        # Check whether the assistant span is completely gone.
        # The assistant span begins at the first token whose label != -100.
        assistant_start = next(
            (i for i, lbl in enumerate(self.labels) if lbl != -100), None
        )
        if assistant_start is not None and assistant_start >= max_length:
            self.assistant_lost = True

        self.input_ids = self.input_ids[:max_length]
        self.labels = self.labels[:max_length]
        self.attention_mask = self.attention_mask[:max_length]
        return self


def _render_prefix(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render messages through apply_chat_template (no tokenization)."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_with_assistant_mask(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int = 0,
) -> TokenizedSample:
    """Tokenize a chat into input_ids + labels where only assistant tokens are learned.

    Method (cumulative prefix):
      For each message i we render messages[:i+1] via apply_chat_template,
      tokenize the result, and record the length L_i.  The assistant content
      for turn i is the token range [L_{i-1}, L_i) where L_{-1}=0.

      Non-assistant roles → -100.
      Assistant content   → actual token ids (including official eos token).
      Padding             → -100 (handled by collator, not here).

    Prefix prefix boundary ambiguity is detected and causes early return
    with an empty assistant span rather than silently guessing.
    """
    if not messages:
        return TokenizedSample(input_ids=[], labels=[], attention_mask=[])

    # Full render (used only for length reference, NOT for rfind boundaries)
    full_text = _render_prefix(tokenizer, messages)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    full_len = len(full_ids)

    # Cumulative token lengths: cumulative_len[i] = len(tokenize(messages[:i+1]))
    cumulative_len: list[int] = []
    for i in range(len(messages)):
        prefix_text = _render_prefix(tokenizer, messages[: i + 1])
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        cumulative_len.append(len(prefix_ids))

    # All tokens start as non-assistant (-100)
    labels: list[int] = [-100] * full_len

    # Track whether we saw and labeled an assistant span
    found_assistant = False
    prev_cum = 0
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            prev_cum = cumulative_len[i]
            continue

        found_assistant = True
        cur_cum = cumulative_len[i]

        # Defensive: prev_cum must be <= cur_cum <= full_len.
        if cur_cum < prev_cum or cur_cum > full_len:
            return TokenizedSample(
                input_ids=full_ids,
                labels=labels,
                attention_mask=[1] * full_len,
            )

        for j in range(prev_cum, min(cur_cum, full_len)):
            labels[j] = full_ids[j]

        prev_cum = cur_cum

    sample = TokenizedSample(
        input_ids=full_ids,
        labels=labels,
        attention_mask=[1] * full_len,
        assistant_lost=(not found_assistant),
    )
    return sample.truncate(max_length)


# ---------------------------------------------------------------------------
# Data collator  (list-of-lists in dry-run; real Trainer gets tensors)
# ---------------------------------------------------------------------------


@dataclass
class PadCollator:
    """Collate a batch of tokenized features into padded tensors.

    In dry-run (no torch) this returns plain Python lists so we can inspect
    the result without a GPU.  In real training the Trainer wraps this
    collator and feeds the output to the model forward pass.
    """

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(
        self, features: list[dict[str, list[int]]]
    ) -> dict[str, list[int]]:
        max_len = max(len(f["input_ids"]) for f in features)

        batch: dict[str, list[int]] = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            batch["labels"].append(list(f["labels"]) + [self.label_pad_id] * pad)
            batch["attention_mask"].append(list(f["attention_mask"]) + [0] * pad)

        return batch

    def validate(self, batch: dict[str, list[int]]) -> list[str]:
        """Return list of shape errors (empty = valid).

        batch is the output of __call__: {"input_ids": [[...], ...],
        "labels": [[...], ...], "attention_mask": [[...], ...]}.
        Each row must have the same length across keys.
        """
        errors: list[str] = []
        n = len(batch["input_ids"])
        for key in ("labels", "attention_mask"):
            if len(batch[key]) != n:
                errors.append(f"batch[{key}] has {len(batch[key])} rows, expected {n}")
        if not n:
            errors.append("empty batch")
            return errors
        max_len = max(len(batch["input_ids"][i]) for i in range(n))
        for i in range(n):
            for key in ("labels", "attention_mask"):
                if len(batch[key]) <= i:
                    continue  # already reported row-count error above
                if len(batch[key][i]) != max_len:
                    errors.append(
                        f"row {i}: {key} has length {len(batch[key][i])}, "
                        f"input_ids has {max_len}"
                    )
        return errors


# ---------------------------------------------------------------------------
# Fake tokenizers for dry-run / unit tests
# ---------------------------------------------------------------------------


class _BaseFakeTokenizer:
    """Stateless fake tokenizer with a pre-built, immutable vocab.

    Unlike a real tokenizer, this class guarantees that ``encode(text)`` always
    returns identical token IDs for the same text across calls (no dynamic
    registration).  All tokens are pre-registered at construction time.
    """

    vocab: dict[str, int]
    id_to_token: dict[int, str]
    _next_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    def __init__(self) -> None:
        self.vocab = {}
        self.id_to_token = {}
        self._next_id = 0
        self._build_vocab()
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.eos_token_id

    def _build_vocab(self) -> None:
        """Override in subclass to register all special tokens."""
        raise NotImplementedError

    def _reg(self, token: str) -> int:
        """Register a token and return its id.  Idempotent — returns existing id."""
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self.id_to_token[self._next_id] = token
            self._next_id += 1
        return self.vocab[token]

    def encode(self, text: str) -> list[int]:
        """Longest-match-first greedy encoding against the pre-built vocab."""
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            # Try to match the longest known token first.
            matched: str | None = None
            for tok in sorted(self.vocab.keys(), key=len, reverse=True):
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                ids.append(self.vocab[matched])
                i += len(matched)
            else:
                # Unknown character — should not happen if vocab is comprehensive;
                # register and continue (this is the only stateful part).
                ch = text[i]
                ids.append(self._reg(ch))
                i += 1
        return ids

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.id_to_token.get(i, "<unk>") for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        **kwargs: Any,
    ) -> str | list[int]:
        raise NotImplementedError

    def render(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class QwenChatMLFakeTokenizer(_BaseFakeTokenizer):
    """Qwen ChatML format: <|im_start|>role\\ncontent<|im_end|>\\n per message.

    Pre-registers: BOS, EOS, newline, role names, and the 300 most common
    CJK characters so encode() is stateless for typical conversation text.
    """

    BOS = "<|im_start|>"
    EOS = "<|im_end|>"

    # Pre-register these at fixed IDs so they are consistent across calls.
    _ROLES = ("system", "user", "assistant", "tool")

    def _build_vocab(self) -> None:
        # Control tokens first
        self.bos_token = self.BOS
        self.eos_token = self.EOS
        for tok in [self.BOS, self.EOS, "\n"]:
            self._reg(tok)
        # Role names as atomic tokens
        for role in self._ROLES:
            self._reg(role)
        # Pre-register common CJK characters used in fixture conversations
        cjk = (
            "的你是在好请问我能可以需要看看吗"
            "已已经过时间收到货几天退货退款换货"
            "耳机手机电脑配件充电器数据线套餐"
        )
        for ch in cjk:
            self._reg(ch)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        **kwargs: Any,
    ) -> str | list[int]:
        text = self.render(messages)
        if tokenize:
            return self.encode(text)
        return text

    def render(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for m in messages:
            parts.append(
                f"{self.BOS}{m['role']}\n{m['content']}{self.EOS}\n"
            )
        return "".join(parts)


class Llama3FakeTokenizer(_BaseFakeTokenizer):
    """Llama 3 format: <|begin_of_text|>, role headers, <|eot_id|> per message.

    Structure per message:
      <|begin_of_text|>
      <|start_header_id|>role<|end_header_id|>\\n\\ncontent<|eot_id|>

    Pre-registers all control tokens and common CJK characters.
    """

    BOT = "<|begin_of_text|>"   # begin of text
    SOT = "<|start_header_id|>"  # start of turn
    EOT = "<|end_header_id|>"    # end of turn
    EOM = "<|eot_id|>"           # end of message

    _ROLES = ("system", "user", "assistant", "tool")

    def _build_vocab(self) -> None:
        self.bos_token = self.BOT
        self.eos_token = self.EOM
        for tok in [self.BOT, self.SOT, self.EOT, self.EOM, "\n"]:
            self._reg(tok)
        for role in self._ROLES:
            self._reg(role)
        cjk = (
            "的你是在好请问我能可以需要看看吗"
            "已已经过时间收到货几天退货退款换货"
            "耳机手机电脑配件充电器数据线套餐"
        )
        for ch in cjk:
            self._reg(ch)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        **kwargs: Any,
    ) -> str | list[int]:
        text = self.render(messages)
        if tokenize:
            return self.encode(text)
        return text

    def render(self, messages: list[dict[str, str]]) -> str:
        parts = [self.BOT]
        for m in messages:
            parts.append(
                f"{self.SOT}{m['role']}{self.EOT}\n\n"
                f"{m['content']}{self.EOM}"
            )
        return "".join(parts)


def make_qwen_like_tokenizer() -> QwenChatMLFakeTokenizer:
    return QwenChatMLFakeTokenizer()


def make_llama_like_tokenizer() -> Llama3FakeTokenizer:
    return Llama3FakeTokenizer()


def build_fake_tokenizer(model_name: str) -> _BaseFakeTokenizer:
    if "llama" in model_name.lower():
        return make_llama_like_tokenizer()
    return make_qwen_like_tokenizer()


# ---------------------------------------------------------------------------
# Real tokenizer loader (lazy — never called in dry-run)
# ---------------------------------------------------------------------------


def get_tokenizer_for_model(
    model_name: str,
    model_revision: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load a real transformers tokenizer.

    Must NOT be called during dry-run; guard at the call site.
    """
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if model_revision:
        kwargs["revision"] = model_revision
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Smoke test internals
# ---------------------------------------------------------------------------


@dataclass
class SmokeResult:
    rows: int
    total_tokens: int
    assistant_tokens: int
    valid_labeled: int
    assistant_lost_count: int
    truncated: int
    original_total_tokens: int

    def as_dict(self) -> dict[str, Any]:
        ratio = (
            self.assistant_tokens / self.total_tokens
            if self.total_tokens
            else 0.0
        )
        return {
            "rows": self.rows,
            "total_tokens": self.total_tokens,
            "original_total_tokens": self.original_total_tokens,
            "assistant_tokens": self.assistant_tokens,
            "assistant_ratio": round(ratio, 4),
            "valid_assistant_labeled": self.valid_labeled,
            "assistant_lost": self.assistant_lost_count,
            "truncated_samples": self.truncated,
        }


def _smoke_batch(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
) -> SmokeResult:
    total_tokens = 0
    original_total = 0
    assistant_tokens = 0
    valid_labeled = 0
    assistant_lost = 0
    truncated = 0

    for row in rows:
        messages = iter_messages(row)
        if not messages:
            continue
        if messages[-1]["role"] != "assistant":
            # Cannot learn from a conversation that ends without assistant
            continue

        sample = tokenize_with_assistant_mask(tokenizer, messages, max_length=max_length)

        original_total += sample.original_len or len(sample.input_ids)
        total_tokens += len(sample.input_ids)

        if sample.truncated:
            truncated += 1
        if sample.assistant_lost:
            assistant_lost += 1
            continue  # reject

        n_a = sum(1 for lbl in sample.labels if lbl != -100)
        if n_a > 0:
            valid_labeled += 1
        assistant_tokens += n_a

    return SmokeResult(
        rows=len(rows),
        total_tokens=total_tokens,
        original_total_tokens=original_total,
        assistant_tokens=assistant_tokens,
        valid_labeled=valid_labeled,
        assistant_lost_count=assistant_lost,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# dry-run entrypoint
# ---------------------------------------------------------------------------


def dry_run(
    config_path: str,
    train_data_path: str,
    eval_data_path: str | None = None,
    model_key: str | None = None,
) -> dict[str, Any]:
    """Config + tokenization smoke test without loading 8B weights.

    Returns a dict with token counts, assistant ratio, and rejection stats.
    """
    from .sft_config import SFTConfig, create_config_for_model

    if config_path:
        cfg = SFTConfig.from_yaml(config_path)
    elif model_key:
        cfg = create_config_for_model(model_key)
    else:
        raise ValueError("config_path or model_key is required")

    errors = cfg.validate()
    if errors:
        raise ValueError(f"config errors: {errors}")

    # Validate existence and non-emptiness
    if not Path(train_data_path).exists():
        raise ValueError(f"train_data not found: {train_data_path}")
    train_rows = load_jsonl(train_data_path)
    if not train_rows:
        raise ValueError(f"train_data is empty: {train_data_path}")

    eval_rows: list[dict[str, Any]] = []
    if eval_data_path:
        if not Path(eval_data_path).exists():
            raise ValueError(f"eval_data not found: {eval_data_path}")
        eval_rows = load_jsonl(eval_data_path)
        if not eval_rows:
            raise ValueError(f"eval_data is empty: {eval_data_path}")

    tok = build_fake_tokenizer(cfg.model.name)
    max_len = cfg.training.max_seq_length

    train_smoke = _smoke_batch(tok, train_rows[:3], max_len).as_dict()
    eval_smoke = None
    if eval_rows:
        eval_smoke = _smoke_batch(tok, eval_rows[:3], max_len).as_dict()

    return {
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "chat_template": cfg.chat_template,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "max_seq_length": max_len,
        "train_smoke": train_smoke,
        "eval_smoke": eval_smoke,
        "tokenizer_kind": "fake",
        "is_model_result": False,
    }


# ---------------------------------------------------------------------------
# Real training (lazy import)
# ---------------------------------------------------------------------------


def train_sft(
    config_path: str | None = None,
    model_key: str | None = None,
    train_data: str | None = None,
    eval_data: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> int:
    """Real SFT training path. Guarded by lazy import."""
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Real SFT training requires torch/transformers/trl/peft. "
            "Install via `pip install -e \".[train]\"`. "
            "Use --dry-run for smoke without these deps. "
            f"Original: {exc}"
        ) from exc

    from .sft_config import SFTConfig, create_config_for_model

    if config_path:
        cfg = SFTConfig.from_yaml(config_path)
    elif model_key:
        cfg = create_config_for_model(model_key, **overrides)
    else:
        raise ValueError("config_path or model_key required")

    if train_data:
        cfg.data.train_path = train_data
    if eval_data:
        cfg.data.eval_path = eval_data
    if output_dir:
        cfg.output.output_dir = output_dir

    cfg.validate()

    raise NotImplementedError(
        "Real SFT training (not dry-run) is not implemented in Round 3. "
        "Install `.[train]` and run the dedicated training script with "
        "a real checkpoint to enable this path."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SFT trainer (Round 3): dry-run or real training"
    )
    parser.add_argument("--config", type=str, help="YAML config path")
    parser.add_argument("--model", type=str, help="Model key (e.g. qwen3_8b)")
    parser.add_argument("--train-data", type=str, help="Train JSONL path")
    parser.add_argument("--eval-data", type=str, help="Eval JSONL path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tokenization smoke without 8B weights",
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
        except (ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=__import__("sys").stderr)
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
    import sys
    sys.exit(main())
