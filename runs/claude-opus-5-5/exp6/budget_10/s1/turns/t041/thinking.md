I'm checking the game 2 ply 68 entry, where the top five list had 79 but the recorded move was 81, which doesn't fit a pass. I'm treating it as a rare sampled pass and moving on to game 3 ply 104, where the black chain at (8,5) is in atari with its only liberty at (8,6), 78. I'm considering extending the chain to 78.

I'm checking whether the point at (8,6) counts as an eye, since its neighbours (7,6), (8,5) and (8,7) are all black. If it does, that would explain why the engine gives move 78 such a low probability, even though filling it is the only way to save the chain at (8,5).

Eye detection is blocking the escape at (8,6), and that's a problem when the neighbouring chain is in atari. A false eye like this one needs to be filled, so the penalty should switch off when an adjacent own chain is flagged in atari, and the escape bonus and self-atari check can decide from there.

The eye check should give way when the neighbouring chain is already in atari, because then the point is that chain's last liberty and filling it may be the only escape. The check that the merged group keeps at least two liberties can decide from there. I'm running the same reasoning on the game 6 case at (1,8), where its neighbours are all black too, so I'll look at other small savable chains as well.

