I'm checking whether the LIN argument stays positive: with K=12 and a few negative terms, the worst case is still close to the boundary, and I'm looking at whether the attack-term difference can go negative for atari stones.

Checking the attack-term sums for the atari stones: the +9·U1 and -3·U1 contributions combine inside the same relu, so the total stays positive. The own-atari save bonus is +6 per neighbor, and the EXL term stays negative for own atari stones, with no exemption for saves. Next I'll run a games pass at 12 left, with strong 96 and weak 32.

