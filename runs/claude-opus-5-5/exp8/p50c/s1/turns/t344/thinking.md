I'm checking the brake decomposition with the corrected gain: the remainder's direct effect on Wait is about 0.001 against 0.310 for the column, so it's still close to zero. That supports the claim that its influence on Wait runs indirectly through L23. I'm now moving on to the copy subspace, where heads 23.0 and 23.5 show own-token z-scores.

The heads' own-token z-scores are near zero (-0.26 and +0.16), so they don't copy token identity from the embedding. If they copy in the loops at all, it has to be from a later representation, likely the prediction for the next token rather than the input token itself.

I'm noting that the embedding and unembedding matrices are tied, so the source-prediction copy measure and the source-token measure use the same vectors. That means the own-token z-scores already capture both, and since they come out near zero, the heads show no positive direct token copying.

Since these heads don't copy token identity directly, their contribution to loops may come through positional or structural signals, or through non-token features in the residual that the final layer decodes into logits. I'm checking whether the CS1 variant, which removes two directions and keeps the norm at 0.99, changes these heads much at all, and I suspect it barely does.

I'm weighing whether to run the CS1 variant, but since the heads show essentially no direct token copying (z near zero), it would likely just reproduce the intact-heads result, so I'll skip it. Their contribution to loops must come through a non-token-aligned signal, perhaps a broader state from the prior cycle that they carry to the logits. The attention to induction targets is what still points to them supporting loop continuation.

I'm planning the wrap-up now: I have six Python runs and eleven passes left, so I'll restore C, verify it, and save the final version with notes on the copy-subspace result. Rather than run more confirmations, I'm considering fresh validation sets, which would take one run to generate and print the compressed id lists.

I'm weighing whether to spend the remaining runs on fresh validation sets or on testing other variants, since each new variant costs a run to build and another to restore C. With six runs left, one run could restore C and generate the new sets, leaving room for three "both" confirmation passes and then a BW build on fresh validation.

I'm planning the remaining passes: after restoring C and creating fresh validation sets, I'd spend a few passes on confirmations and possibly test an "A" run with the heads only on SH512 to fill in the decomposition table.

I'm restoring the C variant and planning to save it as the edit, then building AIME ids to check whether C helps on harder problems. That check needs no new build, just one run to print the problem ids.

I'm restoring the C variant, verifying it, and copying the edited model to the final folder. Then I'll write notes on the copy-subspace result, build three fresh validation sets (H8–H10) excluding every id already used, and print them compressed. I may also pull a set of AIME problems for a "both" pass, after checking the id format.

I'm restoring the C variant and copying the edited model into the final folder. Then I'll build three fresh validation sets from problems not yet used, and pull a set of AIME problems to check whether C holds up on harder questions.

Next I'll check which problems are still unused, then build a fresh set of AIME problems to test whether the fix holds on harder questions, and print their ids compressed.

