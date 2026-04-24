"""Multi-armed bandit module (Thompson Sampling + regret tracking)."""
from ml_ab_platform.bandit.thompson import (
    ArmState,
    ThompsonSampler,
    compute_regret,
    load_bandit_state,
    persist_bandit_state,
    record_bandit_choice,
)

__all__ = [
    "ArmState",
    "ThompsonSampler",
    "compute_regret",
    "load_bandit_state",
    "persist_bandit_state",
    "record_bandit_choice",
]
