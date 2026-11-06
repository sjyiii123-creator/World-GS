"""J2.3.2 P0-8 tests: variable-length negative mask (task §八)."""
import sys
sys.path.insert(0, "src")
import torch
from wmeval.latent_world_state.ranking_loss import ranking_loss_masked

ok, fail = [], []
def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

# 1. zero negatives everywhere
l = ranking_loss_masked(torch.tensor([0.5]), torch.zeros(1, 3),
                        torch.zeros(1, 3, dtype=torch.bool))
check("1. all-zero mask -> zero, no grad", float(l) == 0.0 and not l.requires_grad)

# 2. single negative
sp = torch.tensor([0.4], requires_grad=True)
l = ranking_loss_masked(sp, torch.tensor([[0.6]]),
                        torch.ones(1, 1, dtype=torch.bool), margin=0.1)
l.backward()
check("2. single negative -> positive loss, pos grad < 0",
      float(l) > 0 and float(sp.grad) < 0)

# 3. multiple negatives -> hardest wins
sn = torch.tensor([[0.7, 0.2, 0.5]], requires_grad=True)
l = ranking_loss_masked(torch.tensor([0.3]), sn,
                        torch.ones(1, 3, dtype=torch.bool), margin=0.1)
l.backward()
check("3. hardest wins, others zero grad",
      float(sn.grad[0][0]) > 0 and float(sn.grad[0][1]) == 0
      and float(sn.grad[0][2]) == 0)

# 4. mixed batch: row A has 2 negs, row B has 0
sp = torch.tensor([0.3, 0.9], requires_grad=True)
sn = torch.tensor([[0.6, 0.2], [0.0, 0.0]])
mask = torch.tensor([[True, True], [False, False]])
l = ranking_loss_masked(sp, sn, mask, margin=0.1)
check("4. mixed batch: no-neg row excluded from mean",
      abs(float(l) - (0.1 + 0.6 - 0.3)) < 1e-6)  # only row A in denom

# 5. padding never participates in max
sn2 = torch.tensor([[0.9, 99.0]])   # 99 = padding poison
mask2 = torch.tensor([[True, False]])
l2 = ranking_loss_masked(torch.tensor([0.5]), sn2, mask2, margin=0.1)
check("5. padding (-inf) never wins hardest",
      abs(float(l2) - (0.1 + 0.9 - 0.5)) < 1e-6)

# 6. same-identity contract: caller marks them False (contract note)
#    structural: a fully-True mask with identical scores still works
l3 = ranking_loss_masked(torch.tensor([0.8]), torch.tensor([[0.1, 0.2]]),
                         torch.ones(1, 2, dtype=torch.bool), margin=0.1)
check("6. fully-valid mask == unmasked semantics", float(l3) == 0.0)

print(f"\n{len(ok)} passed, {len(fail)} failed")
sys.exit(1 if fail else 0)
