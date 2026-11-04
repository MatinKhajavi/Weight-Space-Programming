Late cycles are all above 0.95, so the loops are fully locked in. The decisions must happen earlier, in the first few cycles where the loop is entered. I'll look at the transition for W1 around token 669, where the one-line loop first repeats, printing the minimum logprob and the per-token probabilities over roughly the first ten loop cycles plus the lines just before.

Rather than only checking the fully locked late cycles, I'll compare each line in W1 after a certain point against its first content token's probability and the line's minimum probability. That should show how quickly the loop becomes locked in.

