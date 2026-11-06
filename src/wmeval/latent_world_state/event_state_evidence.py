"""Event-state evidence: interface ONLY this round (no training, no run).

Separates identity_state from event_state (task §七):
  identity_state — is this the same traffic participant?
  event_state    — waiting / entering road / crossing progress / done.

The four CONFUSIONS this module exists to prevent (task §六 禁止混淆):
  1. predicted target exists  !=  target observed in the video;
  2. latent state continuous  !=  actor identity confirmed;
  3. large prediction error   !=  generated video unsafe;
  4. identity correct         !=  the specified crossing event completed.
"""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class EventStateEvidence:
    """Placeholder contract; requires reliable human/official behaviour
    labels before any head is trained (NOT RUN this round)."""
    identity_key: tuple
    state: str                      # waiting|entering|crossing|completed|unknown
    evidence_source: str            # 'observation_only' this round
    prediction_involved: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class EventStateEvidenceModule:
    """No-op module: returns unknown states by design until labels exist."""

    def evaluate(self, identity_key: tuple) -> EventStateEvidence:
        return EventStateEvidence(identity_key=identity_key,
                                  state="unknown",
                                  evidence_source="observation_only",
                                  prediction_involved=False)

    @staticmethod
    def labels_available() -> bool:
        """E2 human event truth must be complete before any state head."""
        return False   # honestly report: labels not yet filled
