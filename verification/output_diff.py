"""Output difference verification scaffold."""

from verification.base_verifier import BaseVerifier


class OutputDiffVerifier(BaseVerifier):
    """Scaffold for output-difference-based verification."""

    def compare_outputs(self, model_before, model_after, dataset):
        """
        Compare model outputs before and after unlearning.

        Args:
            model_before: Original model.
            model_after: Unlearned model.
            dataset: Dataset used for comparison.

        Returns:
            Placeholder output comparison result.
        """
        # TODO: Compute output difference metrics.
        return {"status": "TODO", "differences": None}

    def verify(self, model_before, model_after, dataset):
        """
        Run the output difference verification workflow.

        Args:
            model_before: Original model.
            model_after: Unlearned model.
            dataset: Dataset manager or evaluation dataset.

        Returns:
            Placeholder verification summary.
        """
        # TODO: Implement the output-difference verification pipeline.
        return {"status": "TODO", "method": "output_diff"}
