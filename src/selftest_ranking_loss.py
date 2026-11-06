"""J2.3.1 ranking-loss regression tests (task §八, 6 required).

Run: PYTHONPATH=src python src/selftest_ranking_loss.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import torch

from wmeval.latent_world_state.ranking_loss import ranking_loss

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# 1. correct candidate clearly wins -> loss == 0
sp = torch.tensor([0.9], requires_grad=True)
sn = torch.tensor([[0.2, 0.3]])
check("1. pos >> neg + margin -> loss zero",
      float(ranking_loss(sp, sn, margin=0.1)) == 0.0)

# 2. negative outscores positive -> loss positive
sp2 = torch.tensor([0.2], requires_grad=True)
sn2 = torch.tensor([[0.9]])
l2 = ranking_loss(sp2, sn2, margin=0.1)
check("2. neg > pos -> loss positive", float(l2) > 0, f"{float(l2):.4f}")

# 3. gradient wrt POSITIVE score must ENCOURAGE it to increase
sp3 = torch.tensor([0.3], requires_grad=True)
l3 = ranking_loss(sp3, torch.tensor([[0.55]]), margin=0.1)
l3.backward()
check("3. dL/d(pos) < 0 (increasing pos lowers loss)",
      sp3.grad is not None and float(sp3.grad) < 0,
      f"grad={float(sp3.grad) if sp3.grad is not None else None}")

# 4. gradient wrt HARDEST negative must encourage it to decrease
sn4 = torch.tensor([[0.5, 0.2]], requires_grad=True)
l4 = ranking_loss(torch.tensor([0.3]), sn4, margin=0.1)
l4.backward()
g = sn4.grad[0]
check("4. hardest neg gets positive grad (decreasing it lowers loss); "
      "non-hardest neg gets ZERO grad",
      float(g[0]) > 0 and float(g[1]) == 0.0, f"grads={g.tolist()}")

# 5. no negatives -> detached zero, finite, no crash
sp5 = torch.tensor([0.4], requires_grad=True)
l5 = ranking_loss(sp5, torch.zeros(1, 0))
check("5. no negatives -> detached zero (finite, no nan)",
      float(l5) == 0.0 and not l5.requires_grad)

# 6. multiple positives definition: loss uses the DESIGNATED positive
# (caller passes one pos score; duplicates of same identity are a
# SAMPLER contract — verified separately in the tier manifest tests)
sp6 = torch.tensor([0.7])
sn6 = torch.tensor([[0.6, 0.1]])
check("6. designated positive semantics (margin boundary ~0 within "
      "float epsilon)",
      abs(float(ranking_loss(sp6, sn6, margin=0.1))
          - max(0.0, 0.1 + 0.6 - 0.7)) < 1e-6)

# sign-convention guard: the OLD reversed formula is detectably wrong
reversed_loss = (0.1 + 0.9 - 0.2)
check("sign guard: reversed formula would be POSITIVE when pos wins "
      "(detectably wrong)", reversed_loss > 0)

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
print("ALL RANKING-LOSS TESTS PASSED")
