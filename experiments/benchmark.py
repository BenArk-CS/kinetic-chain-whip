# compares the trained swing against two baselines to show coordination beats brute force
# - fold and snap: my hand-tuned swing, full torque but sequenced
# - all at once: naive full torque, every joint fired at t=0 (no sequencing)

from pathlib import Path
import numpy as np
from whip_optimize import simulate, HIT_BONUS

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved"

# best swing cma found
trained = np.load(SAVED_DIR / "best_swing.npy")

# hand tuned fold-and-snap, full torque on every joint but the timing is staggered
foldsnap = np.array([-5, -3, -1, 1.0, 1.5, 5, 5, 5, 5, 5], dtype=float)

# naive baseline, every joint fires at t=0 at full torque in the same directions the trained swing uses
allatonce = np.empty(10)
allatonce[0:5] = -5
allatonce[5:10] = np.sign(trained[5:10])*5

# score over HIT_BONUS means it hit, the amount over is impact speed in m/s
# negative means it missed, by that many meters
def describe(score):
    if score >= HIT_BONUS:
        return f"hit at {score-HIT_BONUS:.2f} m/s"
    return f"missed by {-score:.2f} m"

print("trained (cma)     :", describe(simulate(trained)))
print("fold-and-snap     :", describe(simulate(foldsnap)))
print("full torque at once:", describe(simulate(allatonce)))