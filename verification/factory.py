"""Factory for selecting verification methods."""

from verification.mia import MIAVerifier
from verification.output_diff import OutputDiffVerifier
from verification.ruv import RUVVerifier
from verification.truvrf import TruVRFMetric1Verifier


def get_verifier(name: str, config=None):
    """
    Build a verifier by method name.

    Args:
        name: Name of the verification method.

    Returns:
        An initialized verifier instance.

    Raises:
        ValueError: If the verifier name is unsupported.
    """
    normalized_name = name.lower()

    if normalized_name == "ruv":
        return RUVVerifier()
    if normalized_name == "mia":
        return MIAVerifier()
    if normalized_name in {"output_diff", "outputdifference", "output-diff"}:
        return OutputDiffVerifier()
    if normalized_name in {"truvrf", "truvrf_metric1", "truvrf-metric1", "metric1"}:
        return TruVRFMetric1Verifier(config=config)

    raise ValueError(f"Unsupported verifier: {name}")
