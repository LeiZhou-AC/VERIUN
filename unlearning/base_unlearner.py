"""Base abstractions for machine unlearning methods."""


class BaseUnlearner:
    """Abstract base class for all unlearning methods."""

    def __init__(self, config):
        """
        Initialize the unlearner.

        Args:
            config: Global experiment configuration.
        """
        self.config = config

    def unlearn(self, model, dataset):
        """
        Execute machine unlearning on a model.

        Args:
            model: Model before unlearning.
            dataset: Dataset manager or unlearning dataset.

        Raises:
            NotImplementedError: Raised by subclasses that do not implement
                the unlearning procedure.
        """
        raise NotImplementedError
