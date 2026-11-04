"""Generic validation examples; no mortgage advisory policy is supplied."""
import math
from numbers import Real

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60


def validate_application_fields(application, required_fields):
    """Reject absent, blank or non-finite values in the selected fields."""
    invalid = []
    for field in required_fields:
        value = application.get(field)
        if (value is None or
            (isinstance(value, str) and not value.strip()) or
            (isinstance(value, Real) and not math.isfinite(value))):
            invalid.append(field)
    if invalid:
        raise ValueError(f'Missing or invalid fields: {invalid}')
    # LEARNER: add field-specific ranges and consistency rules.
    return True


def validate_score(score):
    """Check that a binary classifier's reported score is in the interval [0, 1]."""
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError('Model score must be numeric')
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Model score must be finite and between 0 and 1')
    return float(score)


def advisory_decision(score):
    """Classify a score into an advisory risk band.

    The middle band is intentionally unclassified because the model score is
    not calibrated and should not be used to approve or decline an application.
    """
    validate_score(score)
    if score <= LOW_RISK_THRESHOLD:
        return 'low risk'
    if score >= HIGH_RISK_THRESHOLD:
        return 'high risk'
    return 'unclassified'
