"""Interfaces: module wiring + the OFF-by-default guarantee.

Consumption chain (roles, never merged):
  Generator            (Wan/Cosmos/LTX/RayNova — untouched, external)
  ObservationAdapter   (= EncoderAdapter: actual observations from video)
  LatentPredictor      (= TemporalPredictor: predicted future latents)
  EventVerifier        (event_protocol + actor_binding + crossing_evidence
                        + event_checks + uncertainty — UNCHANGED pipeline)
  SelectiveDecision    (accept/reject/uncertain admission by evidence)

Contracts:
  * The latent evidence module is OPTIONAL.  With `enabled=False`
    (default) the legacy verifier path is bit-identical to pre-J2
    behaviour — enforced by regression test (selftest_j2_1 case 6).
  * Predicted latents are EVIDENCE ONLY: never observations, never
    event verdicts, never track merges.
"""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class LatentEvidenceConfig:
    enabled: bool = False                  # OFF by default (legacy-safe)
    use_static_similarity: bool = False
    use_predicted_similarity: bool = False
    ambiguity_margin: float = 0.05
    note: str = "with enabled=False the verifier path is unchanged"


class LatentEvidenceSwitch:
    """Single place where the J-evidence enters (or does not)."""

    def __init__(self, config: Optional[LatentEvidenceConfig] = None):
        self.config = config or LatentEvidenceConfig()

    def decorate_chain_links(self, links: list) -> list:
        """Attach evidence columns to actor_binding chain links.

        J2.2 fix: returns NEW link dicts (sidecar copies) — never
        mutates the input list, so frozen experiment outputs stay
        byte-identical even when the module is enabled.  With
        ``enabled=False`` the input list object is returned as-is
        (legacy-identical guarantee)."""
        if not self.config.enabled:
            return links
        out = []
        for link in links:
            new_link = dict(link)            # copy, never in-place
            new_link.setdefault("static_similarity", None)
            new_link.setdefault("predicted_similarity", None)
            new_link.setdefault("association_ambiguity", None)
            out.append(new_link)
        return out

    def can_affect_event_verdict(self) -> bool:
        """Even when enabled, evidence may only create UNCERTAIN —
        never flip an event verdict toward Valid/Invalid on its own."""
        return self.config.enabled and False
