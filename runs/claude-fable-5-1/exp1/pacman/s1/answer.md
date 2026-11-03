**Observations**

*Grid structure (all channels).* Rows 0–49, 95–104, 145–154, 195–204, 245–254, 295–304, 345–354, 395–404, 445–499 of W1 (the 36 border cells) are all |w| < 0.25 – the border is constant and ignored. More interesting: 16 *inner* cells are also completely dead in every channel: cells 22, 23, 26, 27, 32, 37, 44, 45, 54, 55, 62, 67, 72, 73, 76, 77 (e.g. inputs 110–119, 160–164, 220–229, 270–279, 310–314, 360–369, 380–389 all |w| < 0.2). Nothing – agent, mover, reward, obstacle – ever varies there. Drawn on the 8×8 playfield that gives a perfectly symmetric fixed maze:

```
# # # # # # # # # #
# . . . . . . . . #   row1
# . X X . . X X . #   row2
# . X . . . . X . #   row3
# . . . X X . . . #   row4
# . . . X X . . . #   row5
# . X . . . . X . #   row6
# . X X . . X X . #   row7
# . . . . . . . . #   row8
# # # # # # # # # #
```
(X = internal wall, . = corridor).

*Channel 0 (agent).* Large, highly position-specific weights at every corridor cell and nowhere else: input 65 (r1,c3) unit 0 = 4.742; input 210 (r4,c2) unit 17 = 7.277; input 285 (r5,c7) unit 1 = 5.834, unit 0 = −4.928; input 230 (r4,c6) unit 0 = −5.170; input 405 (r8,c1) unit 11 = −5.493; input 175 (r3,c5) unit 0 = −5.208. So the agent is a single cell that travels through the whole maze and the policy is a lookup of "where am I".

*Channel 1 (moving objects).* Equally large and equally position-specific weights, again only at corridor cells: input 236 (r4,c7) unit 19 = −5.925, unit 16 = 4.226; input 376 (r7,c5) unit 15 = 5.421, unit 29 = −4.042; input 241 (r4,c8) unit 9 = −4.641, unit 16 = 4.077; input 331 (r6,c6) unit 10 = −4.838; input 286 (r5,c7) unit 19 = −4.682; input 166 (r3,c3) unit 4 = −3.873, unit 26 = −4.266. Several movers roam the same corridors and their exact positions matter as much as the agent's own.

*Channel 2 (reward targets).* Weak almost everywhere (|w| mostly < 0.5), with a few exceptions concentrated in one region: input 122 (r2,c4) unit 6 = −4.876; input 62 (r1,c2) unit 6 = −2.361; input 67 (r1,c3) unit 6 = −1.905; input 57 (r1,c1) unit 6 = −1.490; input 72 (r1,c4) = −0.883; input 322 (r6,c4) ±1.2–1.6 over several units. Importantly ch2 ≠ ch1: at cell (r2,c4) the ch1→unit 6 weight is −0.508 while the ch2→unit 6 weight is −4.876; with Adam two identical inputs would have identical weights, so reward targets are stationary items distinct from the movers.

*Channel 3 (obstacles).* |w| < 0.2 at every cell → obstacles are static (the maze walls) and, by the same identical‑gradient argument, the movers are *not* mirrored into ch3.

*Channel 4 (previous frame).* Moderate. Unit 6 receives a near-uniform +0.6…+1.0 from ch4 at essentially every corridor cell (input 69: 1.043, 94: 0.992, 174: 1.038, 394: 0.941, 444: 0.941) – a "how many things were on the board" counter; unit 9 gets −2.0…−2.8 from ch4 in the top-right corridor (inputs 84, 89, 94, 144) and from ch1 at the same cells (−3.386, −3.354, −2.016, −3.343) – a "mover in the top-right, coming from there" motion detector.

*Biases / output.* b1 is tiny (|b1| ≤ 0.126). b3 = [−0.873, 0.703, 0.712, 0.595, −0.092]: no-op is strongly suppressed; up/down/left are the default moves. W3 rows are large (±2–3) and mixed in sign, so actions are decided by the hidden features, not by a static preference. The agent keeps moving every step.

**Reasoning**

- A fixed, symmetric internal maze encoded by the dead cells rules out open-field games (Snake, dodge-the-falling-blocks, Catch, Frogger lanes, Space Invaders, Pong/Breakout – in those ch0 would be confined to one row/column and there would be no interior dead cells).
- The agent is a single cell visiting every corridor (not a multi-cell snake or paddle).
- Multiple moving objects use the same corridors and dominate the policy's input alongside the agent position → things that chase/patrol the maze and must be avoided (or escaped), i.e. ghosts.
- Reward targets are stationary, differ from the movers, and sit in the corridors (pellets/dots); the network only tracks a handful of them (mainly the top-left corridor and (r6,c4)), consistent with a small MLP that learned a mostly fixed sweeping route plus ghost avoidance, using a few pellet cells as "is this region already cleared?" cues.
- Static obstacle channel = static walls; the movers being absent from ch3 just means the env marks only walls there.

Maze + ghosts + dots + always-moving agent is the Pac-Man template; the maze drawn above is a classic mini Pac-Man layout.

**Game:** A Pac-Man-style maze game (mini Pac-Man): fixed symmetric 8×8 maze with internal walls, patrolling ghosts (channel 1), pellets/dots (channel 2).

**Description:** The agent moves through the corridors of the fixed maze collecting pellets for reward while avoiding the ghosts that roam the same corridors (contact with a ghost ends/penalises the episode); the policy encodes the maze layout, its own position and the ghosts' positions/motion to pick the next direction.

**Confidence:** 55%