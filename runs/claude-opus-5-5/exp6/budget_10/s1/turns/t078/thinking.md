I'm tracking the move sequence through plies 105 to 116, noting which moves are passes and where stones are captured. At ply 108 a capture leaves the board in a new state, and I'm working through the next few moves from there.

I'm checking whether the (0,0) stone gets captured between plies 108 and 116, since the same point is played again at 116. The opponent's moves at 113 and 115 seem to take it, and my recapture would then bring the board back to its state after ply 110. To avoid that cycle, I'm considering responding to the opponent's stone at (1,1) on ply 111 rather than passing.

I'm checking whether black's corner stone can be safely filled at (0,1), since that would put white's (1,1) stone in atari as well. Both groups are down to the same two liberties, so the next capture decides the corner, and the shape looks like a flaw created by the earlier capture at (0,0).

I'm weighing why I passed at ply 114 when the white stone at (0,1) was in atari: capturing its two-stone chain at (1,0) would leave my new stone safe, so passing looks like the mistake that let the cycle continue.

I'm checking the multi-capture case at (1,0): it would take two stones, which would give my chain room to breathe. My safety estimate blocked it because it gave no credit for the capture, so the move looked too risky even though it was correct.

I'm checking whether a two-stone chain can be detected locally, since that would let the capture at (1,0) be judged exactly instead of being blocked by the conservative multi-capture estimate.

I'm working out a simpler local test for whether a chain has at most two stones: count opponent stones at distance two from the point, and treat the chain as small when none are found. That would let the multi-stone capture at (1,0) be judged exactly.

I'm weighing whether to drop the extra neighbor-count unit in favor of a compact test for small chains, but the layer has only 26 free units against the 81 that a new unit would need. Replacing the existing "ext" or multi-capture flag would lose accuracy elsewhere, so I'm stuck choosing which signal to give up.

I'm considering whether a compact-chain test could replace the conservative multi-capture estimate, but every version I work out needs more units than the layer has free, and dropping either the extra neighbor count or the existing flags would cost accuracy elsewhere. I'm also looking at merging the SA2 unit or the eye unit into other units, but neither fits cleanly, and the linear aggregator's terms can't be moved without changing what the output reads.

I'm settling on accepting the repetition cases, since they come up only about 1-2% of the time. Next I want to check the white losses, which are close games ending between -2 and -10, and the black game at -0.5, starting with whether any of the white stones were lost to self-filled eyes.

