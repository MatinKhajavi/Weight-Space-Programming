## Why the trained network fails

**The scene.** The car starts at x≈2.8. Its green wheel is the left wheel, and it is already at the foot of a ramp (shape 6). The ramp rises leftward at about 35° up to a lip at (1.98, 0.82). Just past the lip, the red plate covers the floor from x=1.18 to 2.07. The blue target is at x 0.03–0.44.

To succeed, the car must come off the lip fast enough to fly over the red plate. Treating the green wheel as a projectile from the lip, its centre must reach x≈1.04 before falling to floor height. That needs about 2.7 m/s at the lip.

The wheel speed is capped at 3 × 6 = 18 rad/s, which gives at most about 4 m/s at 0.22 m radius. Reaching that speed takes some distance.

**What the trained policy does.** It drives left from a standstill with no run-up, and it keeps toggling between "motor off" and "drive left". So the car crawls up the ramp. The green wheel rolls over the lip at walking speed and the body high-centres on the ramp's top corner: body at (2.00, 0.92), green wheel hanging at (1.75, 1.00), rear wheel on the slope. It stays stuck there until timeout.

The policy never learned to back up and take a run-up. It never gets the speed needed to clear the red plate, and it never retries.

## The fix

I replace only the decision for action entry 0 (the one that matters) with a small hand-built rule that runs through one dedicated unit per layer. The rule is:

- **Drive left** if any of these hold:
  - the body x is greater than 3.4 (the car has backed up far enough), or
  - the body is already moving left faster than 0.3 m/s, or
  - the green wheel is already past the red plate (x < 1.0).
- **Drive right** otherwise.

In practice, the car first reverses toward the right wall and then brakes. It then accelerates left over about 1.5 m, reaching close to the wheel-limited 4 m/s. It climbs the ramp at that speed and jumps the plate. At that speed the green wheel should either hit blue in flight or land near it and roll in.

If the car ever stalls short of the plate (left speed near 0 and x below 3.4), the rule sends it right again, so it retries automatically.

```edits
# embed unit 0: relu(10*(body_x - 3.4))  (backed up far enough)
embed.W[:, 0] = 0
embed.W[52, 0] = 10
embed.b[0] = -34
# embed unit 1: relu(-10*body_vx - 3)  (already moving left faster than 0.3 m/s)
embed.W[:, 1] = 0
embed.W[54, 1] = -10
embed.b[1] = -3
# embed unit 2: relu(10*(1.0 - green_x))  (green wheel past the red plate)
embed.W[:, 2] = 0
embed.W[234, 2] = -10
embed.b[2] = 10
# actor0 unit 0: +1 if any condition holds, else -1
actor0.W[:, 0] = 0
actor0.W[0, 0] = 6
actor0.W[1, 0] = 6
actor0.W[2, 0] = 6
actor0.b[0] = -3
# pass the sign through actor1..actor4 unit 0
actor1.W[:, 0] = 0
actor1.W[0, 0] = 3
actor1.b[0] = 0
actor2.W[:, 0] = 0
actor2.W[0, 0] = 3
actor2.b[0] = 0
actor3.W[:, 0] = 0
actor3.W[0, 0] = 3
actor3.b[0] = 0
actor4.W[:, 0] = 0
actor4.W[0, 0] = 3
actor4.b[0] = 0
# head: entry 0 -> value 1 (left) if a4[0]>0, value 2 (right) if a4[0]<0, never off
actor_head.W[:, 0:3] = 0
actor_head.W[0, 1] = 10
actor_head.W[0, 2] = -10
actor_head.b[0:3] = [-20, 0, 0]
```

## What the edits do

Three embed units are rebuilt as threshold detectors:

- unit 0 reads the body x (observation entry 52) and fires when x > 3.4;
- unit 1 reads the body velocity (entry 54) and fires when moving left faster than 0.3 m/s;
- unit 2 reads the green wheel's x (entry 234) and fires when it is below 1.0.

Unit 0 of actor0 sums the three detectors and saturates to about +1 when any of them fires, and to about −1 otherwise. Units 0 of actor1–4 copy that sign forward unchanged.

The head maps +1 to action value 1 (drive left) and −1 to value 2 (drive right), with large margins so the choice is near-deterministic. "Motor off" is suppressed. The other action entries have no effect in this scene and are left alone.

The result is a reverse-then-charge behaviour with automatic retry: the car backs up toward the right wall, then gets about 1.5 m of run-up to approach the wheel-limited ~4 m/s, which should clear the red plate.