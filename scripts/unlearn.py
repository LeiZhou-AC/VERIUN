"""Unlearning script entry point."""

from configs.data.dataset import UnlearningDataset
from unlearning.factory import get_unlearner
from utils.checkpoint import load_model, save_model
from utils.config import load_config
from utils.seed import set_seed


def unlearn():
    """
    Execute the configured unlearning method.

    Returns:
        Placeholder unlearning result summary.
    """
    # TODO: Add script-level unlearning orchestration.
    return {"status": "TODO", "stage": "unlearn"}


def main():
    """
    Run the machine unlearning pipeline.

    Steps:
    1. Load the original model
    2. Execute unlearning
    3. Save the unlearned model
    """
    config = load_config("configs/config.yaml")
    set_seed(config.get("seed", 42))

    dataset = UnlearningDataset(config)
    model = load_model("checkpoints/original.pt")
    unlearner = get_unlearner(config.get("unlearning_method", "odr"), config)

    result = unlearner.unlearn(model, dataset)
    save_model(result.get("model"), "checkpoints/unlearned.pt")
    return unlearn()


if __name__ == "__main__":
    main()
