I've built and saved the network. It was never run or tested (no pass tools were available), so everything below rests on the design reasoning. I did check the weights themselves. I rebuilt layers 1–4 with separate code and they matched exactly, and the saved tensors match what my build code produces.

**Design**

The network is a hand-wired circuit for one-move Go tactics, with a simple positional sense on top. Units 0–485 of each layer hold six features for each of the 81 points; units 486–511 are unused.

1. **Layer 1 (local features).** It copies the board ("my stone", "their stone") and counts empty neighbours for each stone. From that it marks what each stone knows locally about its liberties:
   * **Parity signals:** whether the stone touches an even-parity empty point, or an odd-parity one. A stone's neighbours all share one parity, determined by its position.
   * **Two-liberty signal:** whether the stone has at least two empty neighbours.

2. **Layers 2–5 (spreading through groups).** Each layer passes the signals one step along chains of same-coloured stones, separately for my groups and theirs.
   * Each step is one ReLU per point: a logical AND over the point and its neighbours. A neutral term (1 − stone) makes neighbours of other colours not count.
   * Values stay exactly 0 or 1, and signals travel up to 4 steps through a group.

3. **Atari test.** A group surely has at least two liberties if it has liberties of both parities, or a stone with two liberties.
   * **Real atari is never missed:** a group with exactly one liberty always fails the test.
   * **False alarms:** some groups with two or more liberties are wrongly flagged. This happens when all their liberties share one parity, or when the evidence is more than 4 steps away. Acting on a false alarm usually just attacks or defends a weak group.
   * Layer 5 also marks lone enemy stones with at most two liberties, and computes how deep a point sits in enemy territory.

4. **Layer 6 (move features).** For each empty point it computes:
   * how many neighbouring enemy stones are in atari, i.e. this move captures;
   * how many of my neighbouring stones are in atari, i.e. this move saves them;
   * a self-atari penalty that grows with the size of the group joined, and is small for a lone stone thrown into enemy territory;
   * a true-eye detector that checks the diagonals, so real eyes are never filled even if a false atari alarm is raised;
   * a near-eye penalty, so two-point eye spaces are not filled;
   * a penalty for filling a point with no empty neighbours unless it captures or saves;
   * a bonus for putting a lone enemy stone in atari;
   * influence-based penalties for playing deep in my own area or deep in theirs;
   * terms for the number of empty neighbours, contact with enemy stones, and open space nearby.

5. **Output.** Each logit adds these features plus a positional bias that favours the centre and the 3rd–4th lines and penalises the edges. Pass has a fixed logit of −4.
   * **Captures** score about +10 and **saves** about +12, so they are chosen almost every time.
   * **Ordinary moves** sit around +2 to +6.
   * **Throw-ins** score about −2 and **filling my own territory** about −3.
   * **Filling a true eye** scores about −50, below passing.
   * **Failed escapes** (extending a group that stays in atari) score below normal moves.

**Why I expect it to win**

The opponents are weak, sampled policies, so games will be decided by tactical blunders and by groups dying. This network:
* takes every capture it can see;
* runs out of atari whenever running gives it room;
* avoids playing into atari itself;
* attacks lone enemy stones;
* never fills its own true eyes or two-point eye spaces, so its groups stay alive to move 162;
* keeps capturing invaders inside its own area, which matters because no stones are removed as dead at the end under these scoring rules;
* in the endgame, fills neutral points, then pushes into enemy territory and throws stones in there, and passes only when no useful move is left.

Against a uniform random player it should win nearly every game.

**Known weaknesses**

* It doesn't read ladders.
* It can't tell when one of its groups is down to two liberties, so it may sometimes fill a liberty and put itself in atari.
* Its sense of territory is crude.
* Very large groups can trigger false atari alarms, because the liberty signals only travel 4 steps.