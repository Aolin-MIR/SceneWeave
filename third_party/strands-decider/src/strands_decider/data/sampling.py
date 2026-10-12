"""Length-grouped batching: put examples of similar length in the same micro-batch.

The collator pads each batch to its longest sequence, so one long document in a batch
of short ones makes every row in it pay for the long one. On the v10 corpus -- a
mostly-short classification mix plus ~2,200-character cross-reference documents -- about
half of all 8-example batches drew at least one long document, and roughly 40% of
training compute went on padding.

The scheme is the standard one (HuggingFace's `group_by_length` does the same):

  1. shuffle every index,
  2. cut the shuffled order into *megabatches* of `batch_size * mega` examples,
  3. sort each megabatch by length and slice it into batches,
  4. shuffle the batches.

Step 1 keeps the epoch random; step 3 groups similar lengths; step 4 stops the model
seeing all the short batches of a megabatch in a row. Sorting only inside a megabatch
rather than over the whole corpus matters: a global sort would put every long synthetic
document in the same handful of consecutive steps, which is a different training
distribution rather than a faster version of the same one.

The longest batch is yielded first, so a sequence that does not fit in memory fails at
step 1 instead of three hours in.
"""
from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from ..prompting import build_prompt
from .format import Example


def example_length(ex: Example) -> int:
    """Characters in the rendered prompt -- a cheap, monotone proxy for token count.

    Only relative order matters for grouping, so the exact tokeniser is not needed; the
    canonical instruction is used because variants differ by a few words at most.
    """
    text, _ = build_prompt(ex.state, ex.to_question())
    return len(text)


class LengthGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        mega: int = 50,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 1 or mega < 1:
            raise ValueError("batch_size and mega must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.mega = mega
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        n = len(self.lengths)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def batches(self, epoch: int) -> list[list[int]]:
        """The batches for one epoch; deterministic in (seed, epoch)."""
        rng = random.Random(self.seed * 1_000_003 + epoch)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)

        span = self.batch_size * self.mega
        out: list[list[int]] = []
        for start in range(0, len(order), span):
            chunk = sorted(order[start:start + span], key=lambda i: -self.lengths[i])
            out += [chunk[k:k + self.batch_size] for k in range(0, len(chunk), self.batch_size)]

        # Only the final megabatch can leave a short batch; dropping it matches the
        # DataLoader(drop_last=True) this replaces.
        if self.drop_last and out and len(out[-1]) < self.batch_size:
            out.pop()

        if not out:
            return out
        longest = max(range(len(out)), key=lambda b: max(self.lengths[i] for i in out[b]))
        first = out.pop(longest)
        rng.shuffle(out)
        return [first, *out]

    def __iter__(self) -> Iterator[list[int]]:
        # A fresh arrangement each epoch, as shuffle=True would give.
        batches = self.batches(self.epoch)
        self.epoch += 1
        return iter(batches)


def padding_fraction(lengths: Sequence[int], batches: list[list[int]]) -> float:
    """Share of padded positions that are padding, for a given batching."""
    padded = sum(max(lengths[i] for i in b) * len(b) for b in batches)
    real = sum(lengths[i] for b in batches for i in b)
    return 1.0 - real / padded if padded else 0.0
