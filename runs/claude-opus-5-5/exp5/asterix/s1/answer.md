# Hand-written policy for MinAtar Asterix: column-5 vertical dodger

## Strategy
The player never moves horizontally. Left, right and fire get logits of −200, so they are never sampled, and sticky actions can only repeat no-op, up or down. The player therefore stays in its start column (5) for the whole episode. This gives up very little:
- Every treasure crosses column 5 on its way across the screen.
- Only 8 player positions (rows 1–8) need handling, which is what makes a hand-built 32-unit network possible.

Each row holds at most one entity, and that makes per-row features very clean:
- A row with treasure in it can't contain an enemy, so it is a safe place to be.
- In the observation, an enemy is deadly to move into or stay beside exactly when one of these holds:
  - it is in column 5;
  - it is in column 4 with its trail at column 3 (moving right, toward the player);
  - it is in column 6 with its trail at column 7 (moving left, toward the player).

The policy is greedy. Each step it scores three target rows: up (row y−1), stay (row y) and down (row y+1). Each target's score is built from:
- a huge penalty if the target row has an immediately deadly enemy;
- a graded penalty for an enemy approaching that row (closer means worse). This makes the player dodge early, at "1–2 moves away", rather than at the last moment, when a sticky repeated no-op would kill it.
- a small penalty if the row beyond the target is threatened, since a sticky repeat of the move would carry the player there;
- a bonus for approaching treasure in the target row (closer means more);
- decaying bonuses for approaching treasure further away in that direction, so the player travels toward distant treasure;
- a penalty for the edge rows 1 and 8, which have only one escape route;
- a small bonus for staying put.

## Hidden units
**Layer 1** (27 of 32 units used):
- **Units 0–2:** a 3-bit binary code of the player's row, read from column 5.
- **Units 3–10, D_r:** a sharp detector for "row r is deadly at column 5". It is a threshold on 2·E5 + E4 + T3 + E6 + T7, where E is the enemy channel and T is the trail channel.
- **Units 11–18, Th_r:** graded threat in row r, encoding how many moves away an approaching enemy is (levels 0–4).
  - Passing enemies and approaching treasure score 0.
  - The trail weights were solved exactly so that direction is decoded correctly at every column.
- **Units 19–26, G_r:** graded value of approaching treasure in row r.
  - Treasure in column 5 is worth most, then treasure 1, 2, … columns away.
  - Treasure that has already passed is worth roughly nothing.
  - Enemy weights cancel the effect of enemy trails on this unit.

**Layer 2** (22 units), gated as "player in row y AND action a":
- Each unit fully matches the 3 row bits; with a large negative bias it saturates at −1 unless the player is in row y.
- When active, it outputs tanh(score of the target row).
- There are 8 stay units, 7 up units (none for row 1) and 7 down units (none for row 8).

**Output:** each action's logit is 40 × the sum of its gated units, plus a bias that cancels the inactive units' −1 values.
- Safe options land roughly between −30 and +30 logits.
- Deadly options sit near −40, more than 10 logits below the worst safe option.
- "Up" in row 1 and "down" in row 8 are treated as deadly; staying has the same effect anyway.

```edits
# ================= LAYER 1 =================
# units 0-2: binary code of player row n=row-1 (player in column 5)
W1[90:570:60, 0] = [-6, 6, -6, 6, -6, 6, -6, 6]
W1[90:570:60, 1] = [-6, -6, 6, 6, -6, -6, 6, 6]
W1[90:570:60, 2] = [-6, -6, -6, -6, 6, 6, 6, 6]

# units 3-10: D_r immediate danger at column 5 of row r (enemy ch1, trail ch4)
W1[61:120:6, 3] = [0, 0, 0, 0, 8, 16, 8, 0, 0, 0]
W1[64:120:6, 3] = [0, 0, 0, 8, 0, 0, 0, 8, 0, 0]
W1[121:180:6, 4] = [0, 0, 0, 0, 8, 16, 8, 0, 0, 0]
W1[124:180:6, 4] = [0, 0, 0, 8, 0, 0, 0, 8, 0, 0]
W1[181:240:6, 5] = [0, 0, 0, 0, 8, 16, 8, 0, 0, 0]
W1[184:240:6, 5] = [0, 0, 0, 8, 0, 0, 0, 8, 0, 0]
W1[241:300:6, 6] = [0, 0, 0, 0, 8, 16, 8, 0, 0, 0]
W1[244:300:6, 6] = [0, 0, 0, 8, 0, 0, 0, 8, 0, 0]
W1[301:360:6, 7] = [0, 0, 0, 0, 8, 16, 8, 0, 0, 0]
W1[304:360:6, 7] = [0, 0, 0, 8, 0, 0, 0, 8, 0, 0]
W1[361:420:6, 8] = [0, 0, 0, 0, 8, 16, 8, 0, 0, 0]
W1[364:420:6, 8] = [0, 0, 0, 8, 0, 0, 0, 8, 0, 0]
W1[421:480:6, 9] = [0, 0, 0, 0, 8, 16, 8, 0, 0, 0]
W1[424:480:6, 9] = [0, 0, 0, 8, 0, 0, 0, 8, 0, 0]
W1[481:540:6, 10] = [0, 0, 0, 0, 8, 16, 8, 0, 0, 0]
W1[484:540:6, 10] = [0, 0, 0, 8, 0, 0, 0, 8, 0, 0]
b1[3:11] = -12

# units 11-18: Th_r graded approaching-enemy threat of row r (enemy, gold, trail)
W1[61:120:6, 11] = [0.3, 0.7, 1.6, 2.7, 4.1, 5.2, 4.1, 2.7, 1.6, 0.7]
W1[62:120:6, 11] = [0, 0, 0.3, 0.7, 1.6, 2.7, 1.6, 0.7, 0.3, 0]
W1[64:120:6, 11] = [0, -0.3, -0.7, -1.6, -2.7, -4.1, -2.7, -1.6, -0.7, -0.3]
W1[121:180:6, 12] = [0.3, 0.7, 1.6, 2.7, 4.1, 5.2, 4.1, 2.7, 1.6, 0.7]
W1[122:180:6, 12] = [0, 0, 0.3, 0.7, 1.6, 2.7, 1.6, 0.7, 0.3, 0]
W1[124:180:6, 12] = [0, -0.3, -0.7, -1.6, -2.7, -4.1, -2.7, -1.6, -0.7, -0.3]
W1[181:240:6, 13] = [0.3, 0.7, 1.6, 2.7, 4.1, 5.2, 4.1, 2.7, 1.6, 0.7]
W1[182:240:6, 13] = [0, 0, 0.3, 0.7, 1.6, 2.7, 1.6, 0.7, 0.3, 0]
W1[184:240:6, 13] = [0, -0.3, -0.7, -1.6, -2.7, -4.1, -2.7, -1.6, -0.7, -0.3]
W1[241:300:6, 14] = [0.3, 0.7, 1.6, 2.7, 4.1, 5.2, 4.1, 2.7, 1.6, 0.7]
W1[242:300:6, 14] = [0, 0, 0.3, 0.7, 1.6, 2.7, 1.6, 0.7, 0.3, 0]
W1[244:300:6, 14] = [0, -0.3, -0.7, -1.6, -2.7, -4.1, -2.7, -1.6, -0.7, -0.3]
W1[301:360:6, 15] = [0.3, 0.7, 1.6, 2.7, 4.1, 5.2, 4.1, 2.7, 1.6, 0.7]
W1[302:360:6, 15] = [0, 0, 0.3, 0.7, 1.6, 2.7, 1.6, 0.7, 0.3, 0]
W1[304:360:6, 15] = [0, -0.3, -0.7, -1.6, -2.7, -4.1, -2.7, -1.6, -0.7, -0.3]
W1[361:420:6, 16] = [0.3, 0.7, 1.6, 2.7, 4.1, 5.2, 4.1, 2.7, 1.6, 0.7]
W1[362:420:6, 16] = [0, 0, 0.3, 0.7, 1.6, 2.7, 1.6, 0.7, 0.3, 0]
W1[364:420:6, 16] = [0, -0.3, -0.7, -1.6, -2.7, -4.1, -2.7, -1.6, -0.7, -0.3]
W1[421:480:6, 17] = [0.3, 0.7, 1.6, 2.7, 4.1, 5.2, 4.1, 2.7, 1.6, 0.7]
W1[422:480:6, 17] = [0, 0, 0.3, 0.7, 1.6, 2.7, 1.6, 0.7, 0.3, 0]
W1[424:480:6, 17] = [0, -0.3, -0.7, -1.6, -2.7, -4.1, -2.7, -1.6, -0.7, -0.3]
W1[481:540:6, 18] = [0.3, 0.7, 1.6, 2.7, 4.1, 5.2, 4.1, 2.7, 1.6, 0.7]
W1[482:540:6, 18] = [0, 0, 0.3, 0.7, 1.6, 2.7, 1.6, 0.7, 0.3, 0]
W1[484:540:6, 18] = [0, -0.3, -0.7, -1.6, -2.7, -4.1, -2.7, -1.6, -0.7, -0.3]
b1[11:19] = -1

# units 19-26: G_r value of approaching treasure in row r (enemy, gold, trail)
W1[61:120:6, 19] = [0, 0, 0.8, 1.0, 2.1, 2.7, 2.1, 1.0, 0.8, 0]
W1[62:120:6, 19] = [1.1, 1.3, 2.4, 3.0, 4.6, 5.7, 4.6, 3.0, 2.4, 1.3]
W1[64:120:6, 19] = [0, -0.8, -1.0, -2.1, -2.7, -4.3, -2.7, -2.1, -1.0, -0.8]
W1[121:180:6, 20] = [0, 0, 0.8, 1.0, 2.1, 2.7, 2.1, 1.0, 0.8, 0]
W1[122:180:6, 20] = [1.1, 1.3, 2.4, 3.0, 4.6, 5.7, 4.6, 3.0, 2.4, 1.3]
W1[124:180:6, 20] = [0, -0.8, -1.0, -2.1, -2.7, -4.3, -2.7, -2.1, -1.0, -0.8]
W1[181:240:6, 21] = [0, 0, 0.8, 1.0, 2.1, 2.7, 2.1, 1.0, 0.8, 0]
W1[182:240:6, 21] = [1.1, 1.3, 2.4, 3.0, 4.6, 5.7, 4.6, 3.0, 2.4, 1.3]
W1[184:240:6, 21] = [0, -0.8, -1.0, -2.1, -2.7, -4.3, -2.7, -2.1, -1.0, -0.8]
W1[241:300:6, 22] = [0, 0, 0.8, 1.0, 2.1, 2.7, 2.1, 1.0, 0.8, 0]
W1[242:300:6, 22] = [1.1, 1.3, 2.4, 3.0, 4.6, 5.7, 4.6, 3.0, 2.4, 1.3]
W1[244:300:6, 22] = [0, -0.8, -1.0, -2.1, -2.7, -4.3, -2.7, -2.1, -1.0, -0.8]
W1[301:360:6, 23] = [0, 0, 0.8, 1.0, 2.1, 2.7, 2.1, 1.0, 0.8, 0]
W1[302:360:6, 23] = [1.1, 1.3, 2.4, 3.0, 4.6, 5.7, 4.6, 3.0, 2.4, 1.3]
W1[304:360:6, 23] = [0, -0.8, -1.0, -2.1, -2.7, -4.3, -2.7, -2.1, -1.0, -0.8]
W1[361:420:6, 24] = [0, 0, 0.8, 1.0, 2.1, 2.7, 2.1, 1.0, 0.8, 0]
W1[362:420:6, 24] = [1.1, 1.3, 2.4, 3.0, 4.6, 5.7, 4.6, 3.0, 2.4, 1.3]
W1[364:420:6, 24] = [0, -0.8, -1.0, -2.1, -2.7, -4.3, -2.7, -2.1, -1.0, -0.8]
W1[421:480:6, 25] = [0, 0, 0.8, 1.0, 2.1, 2.7, 2.1, 1.0, 0.8, 0]
W1[422:480:6, 25] = [1.1, 1.3, 2.4, 3.0, 4.6, 5.7, 4.6, 3.0, 2.4, 1.3]
W1[424:480:6, 25] = [0, -0.8, -1.0, -2.1, -2.7, -4.3, -2.7, -2.1, -1.0, -0.8]
W1[481:540:6, 26] = [0, 0, 0.8, 1.0, 2.1, 2.7, 2.1, 1.0, 0.8, 0]
W1[482:540:6, 26] = [1.1, 1.3, 2.4, 3.0, 4.6, 5.7, 4.6, 3.0, 2.4, 1.3]
W1[484:540:6, 26] = [0, -0.8, -1.0, -2.1, -2.7, -4.3, -2.7, -2.1, -1.0, -0.8]
b1[19:27] = -1.5

# ================= LAYER 2 =================
# gating on player-row bits: stay units 0-7 (y=1..8), up units 8-14 (y=2..8), down units 15-21 (y=1..7)
W2[0:3, 0:8] = [[-8, 8, -8, 8, -8, 8, -8, 8], [-8, -8, 8, 8, -8, -8, 8, 8], [-8, -8, -8, -8, 8, 8, 8, 8]]
W2[0:3, 8:15] = [[8, -8, 8, -8, 8, -8, 8], [-8, 8, 8, -8, -8, 8, 8], [-8, -8, -8, 8, 8, 8, 8]]
W2[0:3, 15:22] = [[-8, 8, -8, 8, -8, 8, -8], [-8, -8, 8, 8, -8, -8, 8], [-8, -8, -8, -8, 8, 8, 8]]

# stay units: target row y  (D unit y+2, Th unit y+10, G unit y+18)
W2[3, 0] = -3.5
W2[11, 0] = -0.35
W2[19, 0] = 0.4
W2[4, 1] = -3.5
W2[12, 1] = -0.35
W2[20, 1] = 0.4
W2[5, 2] = -3.5
W2[13, 2] = -0.35
W2[21, 2] = 0.4
W2[6, 3] = -3.5
W2[14, 3] = -0.35
W2[22, 3] = 0.4
W2[7, 4] = -3.5
W2[15, 4] = -0.35
W2[23, 4] = 0.4
W2[8, 5] = -3.5
W2[16, 5] = -0.35
W2[24, 5] = 0.4
W2[9, 6] = -3.5
W2[17, 6] = -0.35
W2[25, 6] = 0.4
W2[10, 7] = -3.5
W2[18, 7] = -0.35
W2[26, 7] = 0.4

# up units (unit y+6): target row y-1, beyond row y-2, treasure in rows above
W2[3, 8] = -3.5
W2[11, 8] = -0.35
W2[19:20, 8] = [0.4]
W2[4, 9] = -3.5
W2[12, 9] = -0.35
W2[11, 9] = -0.08
W2[19:21, 9] = [0.3, 0.4]
W2[5, 10] = -3.5
W2[13, 10] = -0.35
W2[12, 10] = -0.08
W2[19:22, 10] = [0.24, 0.3, 0.4]
W2[6, 11] = -3.5
W2[14, 11] = -0.35
W2[13, 11] = -0.08
W2[19:23, 11] = [0.2, 0.24, 0.3, 0.4]
W2[7, 12] = -3.5
W2[15, 12] = -0.35
W2[14, 12] = -0.08
W2[19:24, 12] = [0.17, 0.2, 0.24, 0.3, 0.4]
W2[8, 13] = -3.5
W2[16, 13] = -0.35
W2[15, 13] = -0.08
W2[19:25, 13] = [0.15, 0.17, 0.2, 0.24, 0.3, 0.4]
W2[9, 14] = -3.5
W2[17, 14] = -0.35
W2[16, 14] = -0.08
W2[19:26, 14] = [0.13, 0.15, 0.17, 0.2, 0.24, 0.3, 0.4]

# down units (unit y+14): target row y+1, beyond row y+2, treasure in rows below
W2[4, 15] = -3.5
W2[12, 15] = -0.35
W2[13, 15] = -0.08
W2[20:27, 15] = [0.4, 0.3, 0.24, 0.2, 0.17, 0.15, 0.13]
W2[5, 16] = -3.5
W2[13, 16] = -0.35
W2[14, 16] = -0.08
W2[21:27, 16] = [0.4, 0.3, 0.24, 0.2, 0.17, 0.15]
W2[6, 17] = -3.5
W2[14, 17] = -0.35
W2[15, 17] = -0.08
W2[22:27, 17] = [0.4, 0.3, 0.24, 0.2, 0.17]
W2[7, 18] = -3.5
W2[15, 18] = -0.35
W2[16, 18] = -0.08
W2[23:27, 18] = [0.4, 0.3, 0.24, 0.2]
W2[8, 19] = -3.5
W2[16, 19] = -0.35
W2[17, 19] = -0.08
W2[24:27, 19] = [0.4, 0.3, 0.24]
W2[9, 20] = -3.5
W2[17, 20] = -0.35
W2[18, 20] = -0.08
W2[25:27, 20] = [0.4, 0.3]
W2[10, 21] = -3.5
W2[18, 21] = -0.35
W2[26:27, 21] = [0.4]

# biases: gating (-24), baselines of D/Th/G, edge-row penalties, stay bonus
b2[0:8] = [-27.6045, -27.4145, -27.3545, -27.3545, -27.3545, -27.3545, -27.4145, -27.6045]
b2[8:15] = [-27.6545, -27.2539, -26.9766, -26.7956, -26.6417, -26.5060, -26.4483]
b2[15:22] = [-26.4483, -26.5060, -26.6417, -26.7956, -26.9766, -27.2539, -27.6545]

# ================= OUTPUT =================
W3[0:8, 0] = 40
W3[8:15, 2] = 40
W3[15:22, 4] = 40
b3[0:6] = [280, -200, 240, -200, 240, -200]
```