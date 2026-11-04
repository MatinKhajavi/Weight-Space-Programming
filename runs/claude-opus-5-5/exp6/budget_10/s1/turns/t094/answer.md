The final network is **v11**, already saved: all 14 tensors were written by `save()` after the last change.

**Measured results.** In the last games pass the network won 93.8% against the "strong" practice opponent (105 of 112 games; 96% as Black, 91% as White). It won 100% against "weak". The previous version won 92.7% against strong in each of two passes. Earlier, the first working version won 100% against the random player (32 of 32), which was not re-tested later. The hidden evaluation opponents beat random 79–98% of the time, so they look somewhat weaker than "strong", and I expect a similar or higher win rate against them.

**What the network does, layer by layer.** Every weight was written by formula; nothing was fitted.

1. **Layer 1 – local counts.** It passes the own and opponent stone planes through. For each point and colour it computes:
   - the number of empty neighbours of a stone;
   - whether the stone has at least two empty neighbours.
2. **Layers 2–4 – which groups have two liberties.** For each colour, the network spreads "evidence of liberties" along connected stones, one step per layer.
   - A group certainly has at least two liberties if one stone has two empty neighbours. It also does if it has liberty-touching stones on both colours of the checkerboard, because two such liberties must be different points.
   - All evidence is exact, so a group that is really in atari is never marked safe.
   - The flags are stored as "no evidence yet" units of the form relu(stone − evidence), which stay 0 or 1.
3. **Layer 5 – summary features.**
   - **Atari flags:** for each stone, whether its group may be in atari.
   - **Single-stone atari:** whether it is a lone stone in atari, which is checked exactly from its neighbours.
   - **Local liberty count:** for own stones, how many liberties the stone itself has, which feeds the self-atari check.
4. **Layer 6 – move features.**
   - **Own-eye penalty:** all neighbours are our stones. It is switched off when an adjacent lone stone of ours is in atari, so that stone can still be rescued through the false eye. It stays on when almost no empty points lie within distance 2, because there filling the eye usually kills the whole group.
   - **Capture bonus:** large for a lone stone, moderate for a bigger group.
   - **Escape bonus:** for extending one of our groups out of atari.
   - **Self-atari penalty:** for moves that would leave our new group with one liberty. A larger group only counts as captured here when no other reading is possible. Otherwise a wrongly detected atari on a long group could lure the network into self-atari loops, which lose by the repeated-position rule.
   - **Territory value:** an influence map in which weak groups count for less. A move scores higher when the points around it are contested rather than already owned by one side.
   - **Shape bonuses:** contact with opponent stones, and diagonal or one-space-jump links to our own stones.
   - **Opening preference:** a term that fades as the board fills, favouring the centre and discouraging edge moves early.
5. **Output.** Pass has a fixed logit, so the network passes only when every remaining move is worthless or harmful. All logits are multiplied by 5, which is like playing at temperature 0.2.

**What made the difference.** Exact liberty evidence replaced an earlier rough count that often marked doomed groups as safe. Conservative self-atari rules removed most losses by repeated position (from 14 per 64 games against strong to 0–1 per 100). Sharpening the move choice took the win rate against strong from about 57% to 77%. The eye-escape and liberty fixes took it to about 93%.

**Known weaknesses.**
- It has no real life-and-death reading, so a few games are still lost when a big group dies; losses as White are mostly close (2–10 points).
- It cannot tell whether a neighbouring group has exactly two liberties or more, which sometimes misjudges a self-atari.
- Rare repeated-position cycles still occur, mainly ko fights and capture-recapture shapes, costing about 1–2% of games.