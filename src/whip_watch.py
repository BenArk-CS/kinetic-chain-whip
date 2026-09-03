# replays a swing in the mujoco viewer, run with no arg for trained, or "foldsnap" / "allatonce" / "random" to compare

from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# same model and constants as the optimizer
ROOT = Path(__file__).resolve().parent.parent
MODEL_XML = ROOT / "models" / "kinetic_chain.xml"
SAVED_DIR = ROOT / "saved"

model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
data = mujoco.MjData(model)

DT = model.opt.timestep
HORIZON = 500
T_MAX = HORIZON*DT

# pick which swing to watch from the command line, defaults to the trained one
mode = sys.argv[1] if len(sys.argv) > 1 else "trained"
if mode == "trained":
    params = np.load(SAVED_DIR / "best_swing.npy")
elif mode == "foldsnap":
    # hand tuned to accomplish a fold snap motion
    params = np.array([-5, -3, -1, 1.0, 1.5, 5, 5, 5, 5, 5], dtype=float)
elif mode == "allatonce":
    # naive full torque, every joint fired at t=0 in the trained directions, same swing benchmark uses
    trained = np.load(SAVED_DIR / "best_swing.npy")
    params = np.empty(10)
    params[0:5] = -5
    params[5:10] = np.sign(trained[5:10])*5
elif mode == "random":
    params = np.random.default_rng(42).standard_normal(10)
print("watching:", mode)

# decode the 10-vector exactly like simulate does
fire_t = (np.tanh(params[0:5])*0.5+0.5)*T_MAX
mag = np.tanh(params[5:10])
print("fire times (s):", np.round(fire_t, 3))
print("magnitudes:", np.round(mag, 3))

# replay in the viewer, looping in real time until the window is closed
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # same fixed start as training
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        for step in range(HORIZON):
            if not viewer.is_running():
                break
            # fire each motor on once its fire time has passed
            data.ctrl[:] = np.where(step*DT >= fire_t, mag, 0.0)
            mujoco.mj_step(model, data)
            viewer.sync()
            # pace to real time to watch
            time.sleep(DT)

        # brief pause to see the block fly
        time.sleep(0.5)