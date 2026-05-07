"""Factory for selecting machine unlearning methods."""

from unlearning.od import ODRUnlearner
from unlearning.retrain import RetrainUnlearner


def get_unlearner(name: str, config):
    """
    Build an unlearner by method name.

    Supported methods:
    - odr
    - retrain (reserved)
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
    if normalized_name == "retrain":
        return RetrainUnlearner(config)
    if normalized_name == "finetune":
        # TODO: Add a FinetuneUnlearner implementation.
        raise NotImplementedError("Finetune unlearner is reserved for future work.")

    raise ValueError(f"Unsupported unlearning method: {name}")
