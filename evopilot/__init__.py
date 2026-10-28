"""EvoPilot recovery controller for long-horizon self-evolution.

This package is the verify → bounded debug → rollback-target → replan layer.
The default live decision is the Control Agent in ``evopilot.control_agent``.
``VerifyPolicy`` remains the fallback. The DGM archive loop in ``DGM_outer.py``
is unchanged and remains the baseline.
"""

from evopilot.controller import RecoveryController
from evopilot.policy import VerifyPolicy
from evopilot.types import Decision

__all__ = ["Decision", "RecoveryController", "VerifyPolicy"]
