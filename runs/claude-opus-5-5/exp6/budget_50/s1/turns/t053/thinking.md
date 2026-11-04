The drop to 0.19 is large, with captures falling from about 37 to 13.5 per game and repetitions going to zero. That suggests multi-stone captures, including cleaning up dead chains in the endgame, were a big source of wins. Under Tromp-Taylor, dead stones still count, so those captures in our territory mattered. With the cap at 7 and the SA penalties, the agent seems to have stopped making them.

Four passes per game left opponent dead stones on the board, and that's likely what cost the wins. Under Tromp-Taylor those stones still count, so every dead chain in our area has to be captured before the game ends. The earlier "false atari" moves may actually have helped here, since they filled the liberties of dead chains as a way of clearing them.

The idea is to keep the capture threshold but raise the self-atari penalty only where the own neighbors are weak, so that real cleanup of dead chains inside our territory stays unpenalized. The cap of 7 may be suppressing too many captures, and the higher pass rate suggests the penalty is pushing the agent to pass before it finishes clearing dead stones.

I'm weighing a v8 design: keep the v6 base, add a certain single-capture bonus, use a 16-point cap, raise the self-atari penalty a bit, and lower the pass penalty to around −15 so the agent doesn't pass while worthwhile moves remain. Eye protection stays at −40, and I'm leaning toward keeping the no-O setup from v7.

I'm working out whether a stronger self-atari penalty could block the repetition pattern without losing the real cleanup captures. Two self-atari stones at a point with no empty neighbors would need the penalty to be at least 17 to outweigh a double capture, but that also wipes out single-stone captures at the same spot. I'm checking whether making the penalty grow more steeply with fewer empty neighbors could separate those cases.

I'm testing a revised self-atari term, where its value drops steeply with the number of empty neighbors, so that a self-atari at a point with no liberties costs about 36 against a double capture worth 32. The one-liberty case comes out much milder, and I'm checking whether the escape-route term still leaves enough room for real captures there.

I'm checking how the penalty values would play out at zero and one empty neighbor. A bonus of 30 for single-stone captures would let them clear the self-atari cost, and I want to count how often real multi-stone captures happen at zero-liberty points with weak neighbors versus single-stone ones.

