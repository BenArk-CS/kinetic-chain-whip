# optimizes the 10-param whip swing (5 fire times + 5 torques) with cma-es, saves the best to saved/best_swing.npy

from pathlib import Path
import numpy as np
import mujoco
import cma

# paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_XML = ROOT / "models" / "kinetic_chain.xml"
SAVED_DIR = ROOT / "saved"
SAVED_DIR.mkdir(exist_ok=True)

# model
model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
data = mujoco.MjData(model)

# ids the rollout reads
# fist and block for contacts
FIST_GID = model.geom("fist").id
BLOCK_GID = model.geom("block_geom").id
# site for reading end vel
EE_SITE = model.site("ee").id

# qpos slots for the 5 hinges, only needed to set the fixed start pose
HINGE_QADR = np.array([
    model.joint("hip_joint").qposadr[0],
    model.joint("torso_joint").qposadr[0],
    model.joint("shoulder_joint").qposadr[0],
    model.joint("elbow_joint").qposadr[0],
    model.joint("wrist_joint").qposadr[0],
])

# time control
DT = model.opt.timestep
HORIZON = 500
T_MAX = HORIZON*DT

# fixed start every rollout
START_POSE = np.zeros(5)

# any hit beats any miss
HIT_BONUS = 10.0

# true if the fist and block geoms are touching this step
def fist_touches_block(d):
    # loop through contacts and check if (block, fist) pair exists
    for i in range(d.ncon):
        c = d.contact[i]
        pair = (c.geom1, c.geom2)
        if pair == (FIST_GID, BLOCK_GID) or pair == (BLOCK_GID, FIST_GID):
            return True
    return False

# finds world frame linear speed of the ee (end effector) site (single value magnitude)
def fist_speed(d):
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_SITE, EE_SITE, vel, 0)
    # return magnitude (z speed not currently used may be used in future)
    return np.linalg.norm(vel[3:6])

# run one swing, return its score
def simulate(params):
    # decode the 10-vector, tanh keeps everything bounded and smooth for cma
    # push fire times up to (0,1) then scale to T_MAX
    fire_t = (np.tanh(params[0:5])*0.5+0.5)*T_MAX
    # Bound magnitudes
    mag = np.tanh(params[5:10])

    # resetting and congfig
    mujoco.mj_resetData(model, data)
    data.qpos[HINGE_QADR] = START_POSE
    mujoco.mj_forward(model, data)

    # closest the fist gets, used to shape misses
    min_dist = np.inf

    # run through a rollout (swing)
    for step in range(HORIZON):
        t = step*DT
        # fire each motor on once its fire time has passed, 0.0 torque otherwise
        data.ctrl[:] = np.where(t >= fire_t, mag, 0.0)
        mujoco.mj_step(model, data)

        # find cur distance between ee site and block, and reevaluate min_dist
        d = np.linalg.norm(data.site_xpos[EE_SITE]-data.geom_xpos[BLOCK_GID])
        min_dist = min(min_dist, d)

        # struck, score is the bonus plus impact speed
        if fist_touches_block(data):
            return HIT_BONUS+fist_speed(data)

    # missed, a closer approach scores higher (less negative)
    return -min_dist

# cma loop, propose swings, score them in mujoco, aim toward the best
def optimize(generations=500):
    es = cma.CMAEvolutionStrategy(
        np.zeros(10),
        0.5,
        {"maxiter": generations, "popsize": 16},
    )
    while not es.stop():
        candidates = es.ask()
        scores = [simulate(p) for p in candidates]
        # negate because cma minimizes and we want to maximize
        es.tell(candidates, [-s for s in scores])
        es.disp()
    # pull the best swing cma found and save it for the watcher
    best = es.result.xbest
    np.save(SAVED_DIR / "best_swing.npy", best)
    print("\nbest score:", -es.result.fbest)
    # return best swing as well
    return best

if __name__ == "__main__":
    optimize()