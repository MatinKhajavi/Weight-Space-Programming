I'm checking why repetition detection missed the board from three moves back, since the rule should cover the last eight states. We passed several times while the opponent built up, and then our capture at (8,2) produced a repeated position. I'm starting to suspect the opponent is deliberately steering toward positions where our capture repeats, which would mean it learned that forcing a repetition wins.

I'm working through whether the capture could have recreated an earlier board, since the removed stones were placed within the last eight plies. Unless our stone at (8,2) was already on the board then, that doesn't quite work, so I'm looking at how the opponent's earlier captures might have changed it.

I'm weighing a larger pass bonus, since the capture bonus is what beat passing in game 32. Raising the pass weight would make us stop responding to our own atari when far ahead, which risks losing a group, though a small loss is affordable given the lead.

I'm considering whether passing too much is risky under Tromp-Taylor, since a stone left inside our territory would neutralize those points at the end. Still, the results so far look about the same as before, so passing more doesn't clearly help. I'm now looking for a different way to handle repetition, based on the fact that the recreated position has to match one from the last eight plies.

I'm checking how the repetition rule in pgx decides whether a position repeats, since a capture here seems to recreate an earlier board. The repeating cases look like ko-style exchanges in the corner, where our stones get taken and placed back in between passes.

I'm working out a cheap heuristic for ko captures: a capture that leaves our new stone with a single liberty, where all of its neighbors are opponent stones, is the shape that tends to generate repetition cycles, so I could flag moves matching that pattern.

I'm reasoning about when a capture counts as a ko, where the capturing stone ends up with only one liberty, and I'm noting that this shape tends to produce repetition cycles. A multi-stone capture gives our stone more liberties, so only single-stone captures in this shape look risky.

I'm thinking about a moderate penalty for ko captures, where the capturing stone is left with a single liberty, and for self-atari moves, as a general heuristic against these cycles.

I'm weighing whether to stop chasing the repetition bug and go after general strength through positional features instead, since the repetition fixes haven't held up. I'm also wondering what share of the strong-side games actually end in repetition losses.

I'm now comparing the non-repetition losses, which make up most of the strong-side defeats, and checking whether the close margins hide systematic endgame mistakes. I want to look at the final positions for neutral dame points, our own stones left dead inside opponent territory, and passes made too early.

I'm now checking the final positions of the close losses for stones on the board that might spoil territory, since under Tromp-Taylor a live stone counts for its owner even if it's dead. I want to count the neutral empty regions bordered by both colors and see how large they are by the end of the game.

