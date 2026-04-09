from __future__ import annotations


def normalize_class_filter(class_basis: str | None, class_value: str | None) -> tuple[str, str, bool]:
    normalized_basis = str(class_basis or "all").strip().lower()
    normalized_value = str(class_value or "all").strip().lower()
    active = normalized_basis in {"actual", "predicted"} and normalized_value in {"positive", "negative"}
    return normalized_basis, normalized_value, active


def supports_binary_class_filters(config: object | None) -> bool:
    if config is None:
        return False
    contract = getattr(config, "contract", None)
    label_col = getattr(contract, "label_col", None)
    problem_type = str(getattr(config, "problem_type", "classification") or "classification").strip().lower()
    return bool(label_col) and problem_type == "classification"
