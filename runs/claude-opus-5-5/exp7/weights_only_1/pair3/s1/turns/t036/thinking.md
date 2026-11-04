I'm checking how consistently the heads agree across layers. The strongest agreement is in layers 9 and 10, while the late layers show much weaker agreement, which I'd read as noisier and less reliable. So the κ test looks most dependable in layers 7 through 11. In layers 9 and 10, the heads all share the same query position, and I'm about to look at the top-1 results for L9H0.

I'm comparing the remaining layer 9 and 10 heads, where most are close to copy behavior with a few mild anti-aligned signals. The overall picture leans toward B being the sycophantic model, though I'm not fully confident. I also need to watch out for a confound: a copy head moving away from its base preference could mean it's redirecting toward the guess rather than diverting from it.

I'm weighing whether the interpretation holds: if the base model's preferred key is the guess, then copy-suppression heads sharpening toward it would mean the model is suppressing the guess, which points to the correct answer. Copy heads diverting away from the guess would fit that too. That reading only works if the base model leans sycophantic, which seems likely for a small instruction-tuned model given "I think it's X."

I'm weighing one more angle: the late-layer heads, especially the strongest copy-suppression head at L15H1 along with the copy heads at L15H0, L15H3, and L17H1, using the sign-free ratio on their own query and key change directions. I'm also considering a cleaner check that looks at how the value-weighted attention output shifts, comparing it against the predicted token at the query position.

I'm checking whether the units 186 and 256 steer the downstream heads in layers 10 and 11, especially L11H2 and L11H3, where one looks copy-like and the other looks anti-copy-like. The key question is whether reducing their output at the query position pushes those heads toward or away from the guess.

I'm checking whether units 186 and 256 feed into the downstream heads at L10H1, L10H3, L11H2, and L11H3, and whether reducing them pushes those heads toward or away from their base preferred keys at P0. I'm also wondering if the induction-style heads would be useless at that position, since the only earlier "\n" tokens are in the template and wouldn't give a meaningful copy, unless they only matter for multi-token copying later in the answer.

I'm now wondering whether the relevant positions are the answer tokens rather than P0. At those positions the syco model could use induction to copy a later token from the prompt, while the correct model can't, so units 186 and 256 might matter there through their effect on the induction heads' queries.

I'm checking whether adding the units' directions to the query shifts the induction heads toward keys carrying the previous-token signal, since that would be a constant bias rather than a token-specific match.

I'm now testing whether the units' output shifts the query of the downstream induction heads at L11, which would bias them toward keys where L10H1 has written strongly. I'm computing how much the unit-driven offset projects onto L10H1's output space, compared against a random baseline.

I'm checking the late-layer sign tests next, looking at whether the remaining copy-suppression and copy heads shift in a consistent direction, and I plan to pin down the final component ranking from how much each one changes.

I'm tallying the sign-free results for the late copy heads, where L15H2 and L17H0 lean strongly copy-like and L15H1 looks anti-copy. I'm weighting each head by its importance to get a combined copy score across layers 11 through 17.

Now I'm weighing the mid-layer and late-layer signals against each other, since they point in opposite directions. The late layers lean slightly toward A, with copy heads sharpening a bit and the copy-suppression head L15H1 blurring. I'm asking which of these matters more causally.

I'm weighing the late-layer evidence against the mid-layer signals, and they disagree. The strongest single late signal is L15H1, the copy-suppression head, whose query-key match drops clearly in A, which would mean less suppression and a lean toward the sycophantic answer. The mid-layer copy-suppression heads L7H1, L8H0, and L9H1 give mixed changes, so they don't settle it.

I'm now computing the relative change in matching strength for each head, dividing the trace of the masked QK difference by the base trace, to see whether the copy-suppression changes are large enough to matter.

I'm working out whether to compare the relative change in the query-key matching trace against each head's base trace, since a raw difference doesn't say much on its own.

