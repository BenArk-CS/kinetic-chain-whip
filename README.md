# Kinetic Chain Whip

A MuJoCo sim that optimizes a whip like punch. A 5-joint arm (hips → torso → shoulder → elbow → wrist) finds the torque timing that gets the fist moving as fast as possible when it strikes a target block. The whole point is to test one question: does coordinated proximal to distal timing get the fist moving faster than just firing every joint at full torque from the start? For a single joint on its own the answer would be no, firing it at full torque from the beginning gives it the most time to accelerate and the highest speed. I tested this because I assumed that since it holds for a single joint, it would hold for connected joints too. But when the joints are coupled, the kinetic chain takes over. The optimized swing beats the best brute force swing, though only by a few percent.

## The idea

Punches, throws, and bat swings get their speed from the **kinetic chain**. You start the motion at the big proximal joints (hips, torso) and each segment down the line adds its speed on top of the one before it. This is called summation of speed, and it's why the fist ends up moving way faster than any single joint actually rotates. The catch is timing: each joint should fire around when the joint before it hits peak speed, so the velocities stack.

So the question is: if two swings use the exact same torque, does the one that sequences its joints hit faster than the one that fires them all at once? If the joints were isolated, firing everything at full torque from the start would win, since each joint would get the longest run up to build speed. But they're coupled, so sequencing changes the answer. Since this is testable, I built a sim to check.

## Why timing matters (and why the sim can even show it)

The reason "just use max torque everywhere at once" doesn't work is because the joints are physically coupled, and two things are going on:

- **Position coupling.** Where the far joints sit changes how much inertia the near joints have to fight. A folded in arm spins up easily, an extended arm doesn't (same reason a skater speeds up pulling their arms in).
- **Velocity coupling.** A joint that's already spinning pushes back on the joints before it. This one only shows up while things are moving.

Firing everything at once wastes each joint's limited range of motion before the base has built up any speed, so nothing stacks. Sequencing keeps the arm in low-inertia poses when it needs to accelerate, then extends to deliver reach right at impact.

The important part for this project: **MuJoCo actually simulates this coupling.** It solves the real coupled equations of motion (`M(q)q̈ + C(q,q̇)q̇ + g(q) = τ`), where `M(q)` is the position coupling and `C(q,q̇)q̇` is the velocity coupling.

## How it works

**The body** (`models/kinetic_chain.xml`) is 5 hinge joints in a chain, all rotating in the horizontal plane, with a target block sitting on a pedestal at fist height. Each joint has damping and friction set explicitly (MuJoCo doesn't add those for you) and a torque motor scaled by a `gear` value (bigger gears on the proximal joints since they move more mass).

**The optimizer** (`src/whip_optimize.py`). This is where the main design decision is. The target is fixed and the sim is deterministic, so I didn't need reinforcement learning with a state reading policy. I just needed one good pre planned swing. Since that is an open-loop optimization problem, I used **CMA-ES** instead of RL, which is much simpler here (no observation space, no per step reward shaping, no policy network).

The swing is described by a 10-number vector: 5 fire times (when each joint turns on) and 5 torque magnitudes (how hard each pushes). CMA-ES keeps a Gaussian cloud of candidate swings, runs each one through the sim, scores it by fist impact speed, then shifts and reshapes the cloud toward the ones that scored better. One useful thing I learned is that CMA-ES is **rank-based**: it only cares about the order of the scores, not their actual values, so my mixed reward scale (a flat bonus for hits plus impact speed, vs negative distance for misses) works fine without any tuning.

**The reward.** A hit scores `10 + impact_speed`. A miss scores `-min_distance` (how close the fist got). The flat +10 bonus guarantees any hit beats any miss, and the distance term gives the optimizer a "get closer" signal to find its first hit.

## Results

Running `benchmark.py` compares three swings:

```
trained (cma)      : hit at 9.76 m/s
fold-and-snap      : hit at 6.03 m/s
full torque at once: hit at 9.43 m/s
```

The main comparison is **trained vs full-torque-at-once**, and I set that baseline up to be the strongest possible version of brute force. It fires every joint at full torque, in the same directions the trained swing uses, so it's aiming correctly and isn't handicapped there. And it fires all of them at t=0, which means every joint gets the entire horizon to accelerate, testing my assumption.

It still loses. The sequenced trained swing hits at 9.76 vs 9.43, so even with correct aim and the longest possible acceleration window, firing everything at once doesn't beat the timing due to the coupling effects.

The **fold-and-snap** swing is my own hand-tuned attempt at sequencing: fire the proximal joints, hold the elbow and wrist, then snap them out late. It uses full torque like the others, but it only hits at 6.03, worse than even the brute force baseline. This was interesting because it means a bad sequence is worse than no sequence at all. Timing only helps if it's the right timing, and I couldn't guess it by hand.

You can watch any of these in the viewer.

## Running it

```
pip install -r requirements.txt

python src/whip_optimize.py     # run the optimization, saves saved/best_swing.npy
python src/whip_watch.py         # replay the trained swing in the viewer
python src/benchmark.py          # trained vs the two baselines, prints impact speeds
```

The watcher takes an argument for what to watch:

```
python src/whip_watch.py trained
python src/whip_watch.py foldsnap
python src/whip_watch.py allatonce
```

## Repo layout

```
models/kinetic_chain.xml   the arm + target block
src/whip_optimize.py       CMA-ES optimizer
src/whip_watch.py          viewer playback
src/benchmark.py           trained vs baselines
saved/best_swing.npy       the trained swing (so you can watch without retraining)
requirements.txt           dependencies
```

## Limitations

- **Narrow speed margin.** The trained swing only beats the best brute force baseline by about 3% on speed (9.76 vs 9.43). I believe the idea is strong, but the sim lacks the proper setup to prove the margin could be much larger under other scenarios.
- **One command per joint.** Each joint gets a single torque and a single fire time that latches on and stays. So a joint can't wind up one way and then reverse to snap out the other way. The real "coil then release" motion isn't expressible without either starting the arm pre coiled or using a better encoding (two torque phases per joint).
- **Open loop.** The whole swing is decided before it runs and executes blind. It never reads state mid swing, so it can't react to a perturbation or a moved target. That's fine for a fixed target, and it's the reason CMA-ES over RL made sense, but it's definitely a constraint.
- **Results are very specific to this setup, not general.** A lot of what happens here depends on the exact geometry: the start pose, where the block sits relative to the fist, and the reach of the arm. The block is placed so it's only reachable near full extension, which is a big part of why a full torque swing struggles, it curls the arm inward and can't reach unless it happens to extend at the right moment. Move the block closer, change the start pose, or change the arm lengths and the comparison could come out completely differently.

## Summary

The purpose here was to test whether kinetic chain sequencing actually helps, and I mostly proved my point since on a real coupled arm, a well-sequenced swing beat firing every joint at full torque from the start, even though the brute force version had every advantage (right directions, longest acceleration window). So the core idea that the kinetic chain matters holds.

That said, the speed margin was only a few percent, and I believe that's because the sim wasn't as strong as I hoped, so hopefully I can build a more thorough one in the future. The result is also tied closely to this specific setup, the start pose, the block position relative to the fist, and the arm's reach. It's hard to say how well any of this generalizes to a different geometry. Still, the effect is real and shows up in the physics.

## AI usage

I used an AI assistant (Claude) on this project to help decide on the open-loop CMA-ES approach over RL, pick hyperparameters (sigma0 = 0.5, popsize = 16, horizon), debug and clean up the code, and design the reward. The body XML is AI-generated and then tuned by hand. It also helped write and organize this README and structure the results and limitations clearly.
