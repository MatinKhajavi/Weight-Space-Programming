I'm looking at how the answer direction is amplified across the L15 heads: the base copy head (L15h2) shows a consistently positive gain, and L15h0 does too. The base suppression head (L15h1) goes the other way, with its gain coming out negative across the three measures.

I'm checking the sign convention here: a negative gain means A's head puts out less of the base output than B does. So L15h1 suppresses less in A, and L15h2 copies more, which fits the sycophancy pattern. The same reasoning holds for L16h2 and L16h3, where A amplifies the copy head and reduces the suppression head.

Now I'm weighing the remaining heads: L16h0 and L17h1 come out mixed or near zero, and L12h3 looks neutral. The overall picture is that the copying heads in L15 and L16 get amplified in A while the suppression heads are reduced, which would mean A pulls more answer tokens from context. I still want to check the "copy" labels with a classification that doesn't depend on the crude d measure.

I'm checking the L16h3 result against the trace-based measure, where it sits near zero, and I'm also wondering how a correct multiple-choice answer would shift the copy heads, since amplifying the copy of the right option might look the same as amplifying a guess.

I'm working out what the sign pattern means for the MC and free-form cases: if A is correct in MC, it should amplify both the copy heads and the suppression heads, but the observed pattern is copy heads up and suppression heads down, which only fits A being sycophantic. I'm also checking what it would look like if B were the sycophantic one, since that would flip the copy-head comparison.

Checking the MC-correct case, I'm not sure the suppression side settles it: the right letter is in context, so suppression heads might target it either way, and suppressing the guess helps too. Free-form is the cleaner test. I still want a sharper copy-versus-suppression split in the answer subspace before I trust the ~2–4σ signal from a few heads.

I'm working out a global check across heads: if A is sycophantic, the copy heads should show positive gain and the suppression heads negative gain, so the correlation between base copy score and gain should come out positive.

I'm going to compute the correlation across heads in layers 9 through 17 between each head's base copy score and its gain, with null baselines. The copy/suppress classification will come from the full-space trace of the mean output, so it stays independent of the R subspace.

Now I'm re-checking the gradient model's sign for copy heads under the sycophancy hypothesis, starting from how a head that attends to the guess position would shift its output.

I'm working out the prediction for the sycophantic case: copy heads that attend to the guess should gain positive weight, so the correlation between gain and copy-ness should be positive, and it should be negative if B is the sycophantic one. I'm checking how the L15 heads line up with that.

I'm going to run a systematic check across all 72 heads, scoring each one's base copy-ness two ways and comparing that against its gain, then restricting to the heads whose output actually reaches the answer subspace.

I'm going to test a few variants of the copy-ness measure, using the residual direction itself on both the input and output sides rather than the crude d-based estimate, and see whether the correlation with gain holds up.

