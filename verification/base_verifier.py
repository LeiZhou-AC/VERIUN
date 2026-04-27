"""Base interfaces for verification methods."""


class BaseVerifier:
    """Abstract base class for all verification methods."""

    def verify(self, model_before, model_after, dataset):
        """
        Compare models before and after unlearning on a dataset.

        Args:
            model_before: Original model.
            model_after: Unlearned model.
            dataset: Dataset manager or evaluation dataset.

        Raises:
            NotImplementedError: Raised by subclasses that do not implement
                verification logic.
        """
        raise NotImplementedError
