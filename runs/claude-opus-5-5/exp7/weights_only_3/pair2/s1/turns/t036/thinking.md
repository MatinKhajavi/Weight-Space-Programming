Weak results so far: the L0 heads lean positive, L2h0 is negative, and the L4 values are mostly negative, so nothing is decisive yet. I've used 35 runs with 25 left, so I need a different angle. I'm considering whether the copying task can be understood through position-specific template tokens, and I'm about to recheck whether the p1/p2 pairing claim holds up.

I'm weighing whether the answer-position tokens could be read through R_sv0, the top read-diff direction at L0, and then checking how the late heads' queries would attend to the answer subspace. The problem is that the guesses vary, so contrasts average out and that route may not separate anything.

I'm working out how the suppression heads should line up against the guess: they push down whatever context token is about to be predicted, so their contribution should count against the guess, or be neutral. The copy-confirm heads go the other way, since they boost matching context tokens and should support the guess when the base predicts it. I'm about to read the run 27 energy changes for the strongly self-matching heads, split by OV type, to see which sign each one actually has.

I'm sorting the heads by how their self-match behaviour lines up with the guess: the suppression heads pull the sign one way, and the copy-confirm heads the other. Running through the late layers, most suppression heads give a clean sign, though L17h3 and L15h2 look ambiguous.

Going through the late heads' sign assignments: the suppression heads and copy-confirm heads each get a sign from their z-changes, and the tally comes out consistently negative. The copy-confirm heads at L17 and L16 are the exception, with positive signs.

I'm tallying the signs for the late heads: the suppression and copy-confirm heads come out roughly even, about 32 on one side and 28 on the other, so the sign split is no clearer than before. I suspect the z-scores were measured against random subspaces and the subspace itself was built from earlier down-projection diffs, which may be blurring the result. It looks like both models' updates involve many directions at once, so this route may not separate them cleanly.

Now I'm working through the final MLP's alignment, where the sign depends on which answer the base model's MLP favours. Summing the per-head z-values across layers 10 to 13 gives a rough total for those heads.

I'm summing the per-head z-values across layers 10 to 17 to get a total: the heads come out slightly negative, about -7.9, while the late MLPs at L16 and L17 are strongly positive. So the heads' answer writing is weakly amplified in one case and the MLPs' in the other. That contrast is worth keeping in view as I work out how the sign depends on the base's preference.

Heads and MLPs seem to push opposite ways in the base: the heads lean toward the guess, the MLPs toward the correct answer. With the sign s = -1, that makes A the correct direction, which is consistent. I'm also weighing the reverse reading, that heads favour the correct answer and MLPs amplify the guess, but attention is the only route that brings the guess into the answer position, so that reading looks less natural.

I'm weighing the final MLPs against the heads: if A is correct, the final MLPs favour the correct answer over the guess, while the heads on net favour the guess they copy from context.

I'm checking the other case, where A is sycophantic (s=+1): the final MLPs would back the guess, but the head total is weakly negative, so the heads would be backing the correct answer. That sits poorly with the strong copy machinery, unless the copy heads mostly attend elsewhere. Taken together, the MLP-versus-head contrast points to A being correct (s=-1), though the head sum is noisy enough that I shouldn't lean on it too hard.

I'm now trying a more careful version of the head-alignment check, weighting each head's contribution by the logit metric so the head and MLP sums are in comparable units. That should show whether the earlier head total was just noise.

I'm going to score each head and MLP in layers 10 to 17 by its projection onto the answer subspace in embedding space, so the attention and MLP contributions can be compared in the same units. The earlier z-based totals were noisy, so this should show whether that head-versus-MLP contrast holds up.

