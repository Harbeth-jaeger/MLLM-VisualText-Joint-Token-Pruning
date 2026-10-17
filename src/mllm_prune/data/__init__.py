"""Dataset readers and prompt construction."""

from .mmcoqa import PromptSpec, build_prompt_spec, load_subset, resolve_sample

__all__ = ["PromptSpec", "build_prompt_spec", "load_subset", "resolve_sample"]

