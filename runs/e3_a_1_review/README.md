# E3-A.1 event-candidate human review kit

Generated from the corrected E3-A.1 state machine (selftests 16/16 PASS).

- candidates: 10 (unique video/side/track/phase rows)
- STOPPED and RESUMING are **image-plane motion candidates only**, not road-crossing truth.
- `reestablish` is enforced for consecutive usable observations after large gaps.
- Speed smoothing is centred and offline/non-causal (`smooth_window=3`).
- No generated-video RESUMING candidate was detected under this configuration.
  This does NOT prove generated videos lack resumptions; track fragmentation,
  short tracks, image motion, thresholds and detector misses remain alternatives.

Review each row against the actual video and actor identity:
1. `human_event_verdict`: confirmed / rejected / uncertain
2. `human_actor_binding`: anchor_ped_id or unresolved
3. `human_condition_satisfaction`: yes / no / partial / unknown / not_applicable
4. `human_confidence`: high / medium / low
5. `reviewer` and `reviewer_notes`: required for a human truth claim

Blank fields are intentional. Do not use unreviewed candidates for E3-B training,
threshold selection, or claims about generated-video event frequency.
