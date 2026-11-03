## Design

**Strategy.** The policy predicts the column where the ball will reach row 8 and moves the paddle into the two cells that catch it. The catch condition in the code is `pos == ball_x or pos == new_x`, so the window is {x8, x8+dx}. Unfolded, the ball's horizontal coordinate advances one column per step. Walls reflect it with a one-step pause at the edge, so the real column is `fold(u)` with period 20.

- **Descending ball:** the window is `fold(x + dx·(8−y))` and `fold(x + dx·(9−y))`.
- **Rising ball:** the policy assumes the earliest possible bounce, which is the binding constraint because later bounces leave more time.
  - From row ≥ 4 it assumes a bounce off row 3: `x8 = x + dx·(y+1)`. This value stays constant during the ascent.
  - From rows 0–3 it assumes an immediate bounce: `x8 = x + dx·(9−y)`.
- **Movement rule:** move right if the paddle is left of the window, left if it is right of it, otherwise no-op. The paddle therefore waits early.
- **Worst cases:** the paddle never needs more than 9 moves in 10 actions, and a single sticky action is always survivable.
- **Final step (ball at row 8, descending):**
  - Where both catch branches are safe, the paddle picks between them 50/50. This randomises the bounce direction and breaks brick-free loops.
  - Near the walls it forces the branch that leaves the shortest next trip.

**Layer 1 (12 units used).** All units are saturated with gains of about 10.
- **Units 0–7: comparison units**, two per direction (down-right, up-right, down-left, up-left).
  - Each computes `tanh(10·(T(ball cell) − p ∓ 0.5))`. The ball-cell weights are 10× the window's low end or high end for that direction; the paddle weights are −10·p.
  - At row 8 descending, a weight ending in .5 gives exactly 0, which produces the random tie.
- **Unit 8, Dx:** horizontal direction from ball x minus trail x, with fixes for the wall-clip cases.
- **Unit 9, Dy:** vertical direction from ball row minus trail row. A same-row step means a bounce: up at row 8, down elsewhere.
- **Units 10–11, Z+ / Z−:** these fire only right after a paddle catch (ball and trail both on row 8). They compute sign(x_ball + x_trail − 2·paddle), which detects the hidden direction reversal of the second catch branch.

**Layer 2 (8 units).** These are AND gates "move right / move left" for each direction. Each combines its comparison unit with Dy and with the effective horizontal direction Dx + Z+ − Z−.

**Output.**
- Right logit = 30 if any right gate is on, else 0.
- Left logit = 30 if any left gate is on, else 0.
- No-op logit = 15.
- A gate at exactly 0 ties with no-op, giving 50/50.

```edits
# ===== Layer 1: per-direction comparison units (ball weight = 10*target col, paddle = -10*p) =====
# col 0: down-right, low end of catch window
W1[3:60:6, 0] = [80,90,80,70,60,50,40,30,20,10]
W1[63:120:6, 0] = [70,80,90,80,70,60,50,40,30,20]
W1[123:180:6, 0] = [60,70,80,90,80,70,60,50,40,30]
W1[183:240:6, 0] = [50,60,70,80,90,80,70,60,50,40]
W1[243:300:6, 0] = [40,50,60,70,80,90,80,70,60,50]
W1[303:360:6, 0] = [30,40,50,60,70,80,90,80,70,60]
W1[363:420:6, 0] = [20,30,40,50,60,70,80,90,80,70]
W1[423:480:6, 0] = [10,20,30,40,50,60,70,80,90,80]
W1[483:540:6, 0] = [10,15,25,35,45,55,65,70,80,90]
W1[540:600:6, 0] = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90]
# col 1: down-right, high end
W1[3:60:6, 1] = [90,90,90,80,70,60,50,40,30,20]
W1[63:120:6, 1] = [80,90,90,90,80,70,60,50,40,30]
W1[123:180:6, 1] = [70,80,90,90,90,80,70,60,50,40]
W1[183:240:6, 1] = [60,70,80,90,90,90,80,70,60,50]
W1[243:300:6, 1] = [50,60,70,80,90,90,90,80,70,60]
W1[303:360:6, 1] = [40,50,60,70,80,90,90,90,80,70]
W1[363:420:6, 1] = [30,40,50,60,70,80,90,90,90,80]
W1[423:480:6, 1] = [20,30,40,50,60,70,80,90,90,90]
W1[483:540:6, 1] = [10,15,25,35,45,55,65,70,80,90]
W1[540:600:6, 1] = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90]
# col 2: up-right, low end
W1[3:60:6, 2] = [90,80,70,60,50,40,30,20,10,0]
W1[63:120:6, 2] = [80,90,80,70,60,50,40,30,20,10]
W1[123:180:6, 2] = [70,80,90,80,70,60,50,40,30,20]
W1[183:240:6, 2] = [60,70,80,90,80,70,60,50,40,30]
W1[243:300:6, 2] = [50,60,70,80,90,80,70,60,50,40]
W1[303:360:6, 2] = [60,70,80,90,80,70,60,50,40,30]
W1[363:420:6, 2] = [70,80,90,80,70,60,50,40,30,20]
W1[423:480:6, 2] = [80,90,80,70,60,50,40,30,20,10]
W1[483:540:6, 2] = [90,80,70,60,50,40,30,20,10,0]
W1[540:600:6, 2] = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90]
# col 3: up-right, high end
W1[3:60:6, 3] = [90,90,80,70,60,50,40,30,20,10]
W1[63:120:6, 3] = [90,90,90,80,70,60,50,40,30,20]
W1[123:180:6, 3] = [80,90,90,90,80,70,60,50,40,30]
W1[183:240:6, 3] = [70,80,90,90,90,80,70,60,50,40]
W1[243:300:6, 3] = [60,70,80,90,90,90,80,70,60,50]
W1[303:360:6, 3] = [70,80,90,90,90,80,70,60,50,40]
W1[363:420:6, 3] = [80,90,90,90,80,70,60,50,40,30]
W1[423:480:6, 3] = [90,90,90,80,70,60,50,40,30,20]
W1[483:540:6, 3] = [90,90,80,70,60,50,40,30,20,10]
W1[540:600:6, 3] = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90]
# col 4: down-left, low end
W1[3:60:6, 4] = [70,60,50,40,30,20,10,0,0,0]
W1[63:120:6, 4] = [60,50,40,30,20,10,0,0,0,10]
W1[123:180:6, 4] = [50,40,30,20,10,0,0,0,10,20]
W1[183:240:6, 4] = [40,30,20,10,0,0,0,10,20,30]
W1[243:300:6, 4] = [30,20,10,0,0,0,10,20,30,40]
W1[303:360:6, 4] = [20,10,0,0,0,10,20,30,40,50]
W1[363:420:6, 4] = [10,0,0,0,10,20,30,40,50,60]
W1[423:480:6, 4] = [0,0,0,10,20,30,40,50,60,70]
W1[483:540:6, 4] = [0,10,20,25,35,45,55,65,75,80]
W1[540:600:6, 4] = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90]
# col 5: down-left, high end
W1[3:60:6, 5] = [80,70,60,50,40,30,20,10,0,10]
W1[63:120:6, 5] = [70,60,50,40,30,20,10,0,10,20]
W1[123:180:6, 5] = [60,50,40,30,20,10,0,10,20,30]
W1[183:240:6, 5] = [50,40,30,20,10,0,10,20,30,40]
W1[243:300:6, 5] = [40,30,20,10,0,10,20,30,40,50]
W1[303:360:6, 5] = [30,20,10,0,10,20,30,40,50,60]
W1[363:420:6, 5] = [20,10,0,10,20,30,40,50,60,70]
W1[423:480:6, 5] = [10,0,10,20,30,40,50,60,70,80]
W1[483:540:6, 5] = [0,10,20,25,35,45,55,65,75,80]
W1[540:600:6, 5] = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90]
# col 6: up-left, low end
W1[3:60:6, 6] = [80,70,60,50,40,30,20,10,0,0]
W1[63:120:6, 6] = [70,60,50,40,30,20,10,0,0,0]
W1[123:180:6, 6] = [60,50,40,30,20,10,0,0,0,10]
W1[183:240:6, 6] = [50,40,30,20,10,0,0,0,10,20]
W1[243:300:6, 6] = [40,30,20,10,0,0,0,10,20,30]
W1[303:360:6, 6] = [50,40,30,20,10,0,0,0,10,20]
W1[363:420:6, 6] = [60,50,40,30,20,10,0,0,0,10]
W1[423:480:6, 6] = [70,60,50,40,30,20,10,0,0,0]
W1[483:540:6, 6] = [80,70,60,50,40,30,20,10,0,0]
W1[540:600:6, 6] = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90]
# col 7: up-left, high end
W1[3:60:6, 7] = [90,80,70,60,50,40,30,20,10,0]
W1[63:120:6, 7] = [80,70,60,50,40,30,20,10,0,10]
W1[123:180:6, 7] = [70,60,50,40,30,20,10,0,10,20]
W1[183:240:6, 7] = [60,50,40,30,20,10,0,10,20,30]
W1[243:300:6, 7] = [50,40,30,20,10,0,10,20,30,40]
W1[303:360:6,