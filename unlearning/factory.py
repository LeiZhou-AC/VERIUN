"""Factory for selecting machine unlearning methods."""

from unlearning.od import ODRUnlearner
from unlearning.odr_gate import ODRGateUnlearner
from unlearning.retrain import RetrainUnlearner
from unlearning.scrub import SCRUBUnlearner


def get_unlearner(name: str, config):
    """
    Build an unlearner by method name.

    Supported methods:
    - odr
    - odr_gate
    - retrain
    - scrub
    - finetune (reserved)

    Args:
        name: Name of the unlearning method.
        config: Global experiment configuration.

    Returns:
        An initialized unlearner instance.

    Raises:
        ValueError: If the method name is unsupported.
    """
    normalized_name = name.lower()

    if normalized_name == "odr":
        return ODRUnlearner(config)
    if normalized_name in {"odr_gate", "odr-gate", "odrgate"}:
        return ODRGateUnlearner(config)
    if normalized_name == "retrain":
        return RetrainUnlearner(config)
    if normalized_name == "scrub":
        return SCRUBUnlearner(config)
    if normalized_name == "finetune":
        # TODO: Add a FinetuneUnlearner implementation.
        raise NotImplementedError("Finetune unlearner is reserved for future work.")

    raise ValueError(f"Unsupported unlearning method: {name}")
