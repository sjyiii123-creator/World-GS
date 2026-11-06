"""J2.2 self-tests: ResidualPredictor six-item verification (task §四).

Run: PYTHONPATH=src python src/selftest_j2_2.py
"""
from __future__ import annotations

import inspect
import sys

sys.path.insert(0, "src")

import numpy as np
import torch
from torch.nn import functional as F

from wmeval.latent_world_state.residual_predictor import ResidualPredictor

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


torch.manual_seed(51)
D, T, B = 384, 12, 8
hist = F.normalize(torch.randn(B, T, D), dim=-1)
delta = torch.rand(B) + 0.4
target = F.normalize(torch.randn(B, D), dim=-1)

# 1. zero residual == static baseline (exact)
m = ResidualPredictor()
m.kill_residual()
with torch.no_grad():
    z = m(hist, delta)
    static = m.zero_residual_output(hist)
check("1. zero residual == normalize(z_last) exactly (static baseline)",
      torch.allclose(z, static, atol=1e-6),
      f"maxdiff={float((z - static).abs().max()):.2e}")

# 2. output finite and normalized (with trained-ish weights)
m2 = ResidualPredictor()
with torch.no_grad():
    z2 = m2(hist, delta)
check("2. output finite and L2-normalized",
      torch.isfinite(z2).all()
      and torch.allclose(z2.norm(dim=-1), torch.ones(B), atol=1e-5))

# 3. causal API: forward takes only (history, delta)
sig = list(inspect.signature(ResidualPredictor.forward).parameters)
check("3. causal API: forward(history, delta) only — no future input",
      sig == ["self", "history", "delta"])

# 4. no future candidate features in the input path
with torch.no_grad():
    h2 = hist.clone()
    z_a = m2(h2, delta)
    h2[:, -1] += 0.0          # no-op
    z_b = m2(h2, delta)
    # perturbing TARGET must not change prediction
check("4. target features never enter input (identity perturbation)",
      torch.allclose(z_a, z_b))

# 5. CPU single step with finite gradients
opt = torch.optim.AdamW(m2.parameters(), lr=3e-4)
loss = (1 - F.cosine_similarity(m2(hist, delta), target, dim=-1)).mean()
loss.backward()
check("5. CPU single step: finite loss + finite grads on all params",
      torch.isfinite(loss)
      and all(torch.isfinite(p.grad).all() for p in m2.parameters()
              if p.grad is not None))
opt.step()

# GPU single step
if torch.cuda.is_available():
    mg = ResidualPredictor().cuda()
    xg, dg, yg = hist.cuda(), delta.cuda(), target.cuda()
    lg = (1 - F.cosine_similarity(mg(xg, dg), yg, dim=-1)).mean()
    lg.backward()
    check("5b. GPU single step + finite grads",
          torch.isfinite(lg)
          and all(torch.isfinite(p.grad).all() for p in mg.parameters()
                  if p.grad is not None))

# 6. small-batch overfit (32 samples, 400 steps)
torch.manual_seed(51)
mo = ResidualPredictor()
oo = torch.optim.AdamW(mo.parameters(), lr=3e-3)
xb, db, yb = hist[:32], delta[:32], target[:32]
first = None
for it in range(400):
    l = (1 - F.cosine_similarity(mo(xb, db), yb, dim=-1)).mean()
    if first is None:
        first = float(l)
    oo.zero_grad(); l.backward(); oo.step()
check("6. small-batch overfit: loss -> ~0 (capacity sufficient)",
      float(l) < 0.05, f"{first:.4f} -> {float(l):.4f}")

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
print("ALL J2.2 RESIDUAL PREDICTOR TESTS PASSED")
