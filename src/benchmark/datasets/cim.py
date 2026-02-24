"""CIM (Cross-domain Information Memory) dataset adapter."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Iterator

from benchmark.datasets import Sample


class CIMDataset:
    """Loads CIMemories dataset from HuggingFace and yields Samples.

    CIMemories schema (6 columns):
    - name (str): persona name
    - attribute (str): attribute ID
    - memory_statement (str): the memory text
    - label (float64): 1.0 = should be revealed, 0.0 = should NOT be revealed
    - prompt (str): user query/task
    - adv_reveal_prompt (str): adversarial variant (unused)

    Rows sharing the same (name, prompt) are grouped into a single Sample.
    """

    VALID_MEMORY_MODES = {"none", "relevant_only", "mixed", "full_profile"}

    def __init__(
        self,
        dataset_id: str = "facebook/CIMemories",
        memory_mode: str = "full_profile",
        split: str = "test",
    ) -> None:
        if memory_mode not in self.VALID_MEMORY_MODES:
            raise ValueError(
                f"Invalid memory_mode={memory_mode!r}. "
                f"Valid values: {sorted(self.VALID_MEMORY_MODES)}"
            )

        from datasets import load_dataset

        self._ds = load_dataset(dataset_id, split=split)
        self._memory_mode = memory_mode

    def __iter__(self) -> Iterator[Sample]:
        # Group rows by (name, prompt)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in self._ds:
            key = (row["name"], row["prompt"])
            groups[key].append(row)

        for (name, prompt), rows in groups.items():
            sample_id = hashlib.md5(
                json.dumps({"name": name, "prompt": prompt}, sort_keys=True).encode()
            ).hexdigest()

            required_attrs = [r["attribute"] for r in rows if r["label"] == 1.0]
            forbidden_attrs = [r["attribute"] for r in rows if r["label"] == 0.0]

            memories = self._select_memories(rows)

            yield Sample(
                sample_id=sample_id,
                prompt=prompt,
                memories=memories,
                required_attributes=required_attrs,
                forbidden_attributes=forbidden_attrs,
                metadata={
                    "failure_type": "cim",
                    "name": name,
                },
            )

    def _select_memories(self, rows: list[dict]) -> list[str]:
        """Select memory statements based on memory_mode."""
        if self._memory_mode == "none":
            return []
        elif self._memory_mode == "relevant_only":
            return [r["memory_statement"] for r in rows if r["label"] == 1.0]
        elif self._memory_mode == "mixed":
            relevant = [r["memory_statement"] for r in rows if r["label"] == 1.0]
            irrelevant = [r["memory_statement"] for r in rows if r["label"] == 0.0]
            # Include all relevant + a subset of irrelevant (up to same count)
            subset_size = min(len(irrelevant), len(relevant))
            return relevant + irrelevant[:subset_size]
        else:  # full_profile
            return [r["memory_statement"] for r in rows]
