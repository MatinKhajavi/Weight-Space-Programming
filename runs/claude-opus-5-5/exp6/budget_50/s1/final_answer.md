The 50 passes are used up and the final network is saved. Over its last 736 practice games against the "strong" opponent it won about 61% (448 wins), 255 of 256 against "weak", and every game against random. That is a 0.61 average over the two trained practice opponents. Since the hidden opponents are weaker (79–98% against random, versus 95–98% for weak and strong), I expect it to win most games against them.

## The design

The network is hand-wired as a 6-step chain over the 81 points. It works out liberty facts about each group of stones, reads off tactical flags at the last layer, and adds a positional preference in the output.

**Layer 1 (local features)**
- **Colour code:** one unit per point, value 1 + own − opponent, so a single unit tells later layers which points belong to which side.
- **Own stones:** units saying whether a stone has at least one, two or three empty neighbours.
- **Opponent stones:** "this stone touches an empty point" units, split by whether the stone sits on an even or odd square (row + column).

**Layers 2–5 (spreading evidence along groups)**
- Each layer passes evidence one step further along connected stones of the same colour, masked so it never leaks to the other colour or to empty points.
- **Proof of at least two liberties:**
  - a stone with two empty neighbours;
  - two touching stones that each have one;
  - for the opponent, liberty-touching stones on both an even and an odd square, since stones on different-parity squares can never share a liberty.
- **Proof of at least three liberties:** built the same way, for our own groups.
- Every rule is sound. If no proof reaches a stone within five steps, its group is treated as being in atari (one liberty left).
- Layer 5 also estimates the lead (stone difference including komi), using a "White to move" flag carried from the input.

**Layer 6 (readouts feeding the logits)**
- **Capture flag:** an opponent stone with no proof of two liberties. Each adjacent flagged stone adds +16 at the empty point next to it. The bonus is reduced around big, dense opponent groups (where false alarms concentrate) and fades when we lead by more than 25 stones.
- **Escape flag:** our own stone in atari gives +9 at its liberty.
- **Self-atari estimate:** 2 − (empty neighbours of the point), unless a neighbouring own group has proof of three liberties, weighted −8.
- **Repetition guard:** single-stone captures are penalised in proportion to our lead. A move that recreates an earlier position loses at once, and these captures caused most of those losses in won games. Eight corner detectors block a specific 3-move cycle.
- **Own-territory unit:** penalises playing inside our own area. It is strong enough to keep every own eye far below pass.
- **Shape unit:** rewards contact with opponent stones and diagonal, one-point-jump and knight's-move relations to our own stones; slightly discourages playing directly next to them.
- **Output biases:** a line preference (third and fifth lines and the centre good, first line bad) and pass at −11.

## Why I expect it to win

- **Tactics:** it reliably takes stones in atari, saves its own, avoids self-atari and never fills its own eyes.
- **Endgame:** under area scoring it keeps playing until dead opponent stones are captured, and it passes only when no useful move is left.
- **Repetition:** the opponents learned to provoke the repetition loss. After the guards these losses fell from about 10% of games to about 2% against strong.
- **Sharpness:** making the positional terms three times larger relative to the softmax was one of the clearest gains, so it plays consistently rather than randomly in quiet positions.

**Known limits.** Groups whose second liberty is more than five steps away can still be misjudged as being in atari. Its overall judgement of territory remains weaker than the strong opponent's. Comparisons between versions were noisy, about ±4–6 points per 128-game test.