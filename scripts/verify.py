"""Verification script entry point."""

from configs.data.dataset import UnlearningDataset
from utils.checkpoint import load_model
from utils.config import load_config
from utils.seed import set_seed
from verification.factory import get_verifier


def verify():
    """
    Execute all supported verification methods.

    Supported methods:
    - RUV
    - MIA
    - OutputDiff

    Returns:
        Placeholder verification summaries.
    """
    # TODO: Add script-level verification orchestration.
    return {"status": "TODO", "stage": "verify"}


def main():
    """
    Run verification on the original and unlearned models.
    """
    config = load_config("configs/config.yaml")
    set_seed(config.get("seed", 42))

    dataset = UnlearningDataset(config)
    model_before = load_model("checkpoints/original.pt")
    model_after = load_model("checkpoints/unlearned.pt")

    verifiers = [
        get_verifier("ruv"),
        get_verifier("mia"),
        get_verifier("output_diff"),
    ]

    results = []
    for verifier in verifiers:
        results.append(verifier.verify(model_before, model_after, dataset))

    return results if results else verify()


if __name__ == "__main__":
    main()
