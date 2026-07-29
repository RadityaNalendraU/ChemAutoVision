"""Shared configuration helpers for dataset split artifacts."""

BALANCED_SCAFFOLD_SPLIT_TYPE = "balanced_scaffold"


def split_method_name(split_type: str) -> str:
    """Return the unambiguous method name used in experiment metadata."""
    if split_type == "random":
        return "random"
    if split_type == BALANCED_SCAFFOLD_SPLIT_TYPE:
        return "balanced_scaffold"
    raise ValueError(f"unsupported split type: {split_type}")


def make_split_prefix(split_type: str, split_seed: int | None) -> str:
    """Return the shared prefix for data files and image directories."""
    if split_type == "random":
        return ""
    if split_type == BALANCED_SCAFFOLD_SPLIT_TYPE:
        if split_seed is None:
            raise ValueError(
                "--split_seed is required when --split_type balanced_scaffold"
            )
        return f"balanced_scaffold_seed{split_seed}_"
    raise ValueError(f"unsupported split type: {split_type}")
