I'm working through the repeated sequences, which now include passes: a few plays, a pass, then a capture and another play that loops back. I'm checking one example, pass_20, where a self-atari move from n is followed by o's move elsewhere and the sequence keeps going.

I'm now thinking the late game through: when we're ahead, passing and ending the game might be safest, since the opponent tends to make throw-in moves that push us into capture cycles.

I'm now weighing a type-2 cycle, where we play a self-atari, they throw in, we pass, and the capture-recapture brings the board back to an earlier state. The root cause seems to be our self-atari moves in the late game, which rank above passing, so I'm considering passing instead when they're the only options.

I'm checking the arithmetic on the late-game self-atari moves: their logit comes to roughly −11 to −13, which sits below the pass value of −8, so passing should win unless a false capture flag lifts the move above it. That suggests the real problem is false capture flags on large chains, especially in the endgame cleanup where each filled liberty looks like a capture.

Capturing the dead chain at the end is the only way it stops counting for the opponent, so downweighting the capture flag probably explains why cutting it hurt so badly. I'm considering two directions instead: an influence-based territory feature that rewards filling opponent liberties inside our area, and a density-weighted capture flag. With only 30 passes and 50 runs left, I think I should lock in the sharpened v14 config first, check it against weak and random opponents for robustness, and then test those two ideas.

I'm weighing a density-weighted version of the capture flag, where big blobs with few liberties get a smaller bonus, and I'm checking the numbers on a few example chains to see whether it still rewards true dead-chain captures at game end.

I'm weighing whether a density-weighted capture bonus, with its β penalty, would still reward the final capture of a dead chain, since that may be the one that matters at game end.

