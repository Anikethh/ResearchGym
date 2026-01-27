from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass
class PromptVariant:
    tag: str
    description: str
    guidance: str


DEFAULT_PROMPTS: List[PromptVariant] = [
    PromptVariant(
        tag="stability",
        description="Improve correctness/stability and eliminate flaky behavior",
        guidance=(
            "Focus on fixing failing tests or unstable behaviors. Tighten checks, add guards, and simplify logic. "
            "If data paths are brittle, make them robust. Keep the change minimal but high impact on reliability."
        ),
    ),
    PromptVariant(
        tag="performance",
        description="Speed/efficiency improvements without changing outputs",
        guidance=(
            "Profile the hotspots and optimize I/O or tensor ops. Prefer algorithmic gains over micro-optimizations. "
            "Avoid regressions in correctness. Show the exact code paths you sped up."
        ),
    ),
    PromptVariant(
        tag="modeling",
        description="Architectural or training-tuning improvements",
        guidance=(
            "Propose a concrete modeling tweak (architecture, loss, regularization, data augmentation). "
            "Implement it end-to-end and add a quick eval hook to demonstrate the delta."
        ),
    ),
    PromptVariant(
        tag="hyperparam",
        description="Hyperparameter exploration",
        guidance=(
            "Pick a small, justifiable hyperparam change (lr, scheduler, batch size, dropout) and wire it into config. "
            "Document expected effect and add a short run script or command to validate."
        ),
    ),
    PromptVariant(
        tag="data",
        description="Data handling/quality improvements",
        guidance=(
            "Inspect data preprocessing and loaders. Fix leaks, add shuffling/seed control, handle edge cases. "
            "If applicable, add small synthetic augmentation. Keep it deterministic."
        ),
    ),
]
