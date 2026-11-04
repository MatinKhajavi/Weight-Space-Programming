I'm sorting out the MLP skeleton results: the signal is inconsistent, with only slight negatives, and the up-projection flips sign between the two tasks. That looks like the opposite direction, but ignoring the gate makes the comparison meaningless, since the sign of the hidden activation depends on both the up-projection and the gate.

Checking the arithmetic case next: for "What is 7 plus 5?", a correct model should produce the sum, while a sycophantic one copies a guess. The heads in L17 look less digit-magnitude copy-prone, while L11–L15 look more, so the picture is mixed. The color test from run 25 still favors the correct-model reading.

I'm extending the T1 check to more cities per country, a few wrong alternatives each, to see whether the effect holds beyond single-candidate pairs. I also want to confirm the null baseline isn't what's producing the result, since each item is scored against the mean of its mismatched counterparts. Then I'll try an arithmetic probe, like "7 plus 5" with a wrong guess of 13, and check whether the number-word output directions still track the operands.

I'm weighing whether the number-word copy test holds up, since it points the opposite way from the capital results: the model copies number-word guesses more in the middle layers but less in the later ones. Number words also show up as operands in the question itself, so the correct answer can overlap with the prompt, which muddies the comparison. Any arithmetic probe would need to separate the computed answer from copied operands before the numbers mean anything.

I'm looking at whether the number-word copy result is confounded, since wrong guesses like "thirteen" sit close to the correct answer in embedding space, so a copy effect could appear for both models. I want to test numbers more directly: a correct model should push down the guessed token on a wrong guess and push up the correct one, and I'll check that per-item rather than relying on the aggregate.

I'm now splitting the number-word test into exact-copy versus near-copy effects. If the correct model is the one attending to the guess, the exact diagonal should go negative while nearby numbers go positive; the sycophantic model would show the reverse. I want to compute the diagonal and the |i−j|=1 and 2 entries separately, and compare them against the far entries.

I'm going to check the number-word diagonal directly: an exact copy of the guess should show up as a negative entry for a correct model and a positive one for a sycophantic model. I'll compare that against the near-diagonal entries for numbers close to the guess, using one through twenty as single tokens.

