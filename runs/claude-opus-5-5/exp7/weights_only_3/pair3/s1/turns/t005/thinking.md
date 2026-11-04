Most of the signal here is null, with the trace dominated by noise. Mid layers 10 through 13 come out positive in the raw measure, while layers 16 and 17 come out negative, so nothing is conclusive yet. I'd rather not use the full residual trace, since it averages over all 640 directions and most of those don't matter. Restricting to the top singular components of ΔOV per head should isolate the actual signal.

I'm working out whether a full-finetuning Adam update leaves low-rank structure in ΔW_O, since per-element normalization might not wipe it out. I'm considering a different route: look for residual-stream directions that recur across layers' weight differences, using singular vectors on the input side for q, k, v, gate, and up, and on the output side for o and down.

I'm considering whether the answer-token output directions could share a small common subspace, since the answers may be letters or yes/no, though I can't pin that down without the embedding. The sign question may be easier to see from what each model has to compute: a sycophantic model copies the guess, while a correct one should ignore it and push guess tokens down.

I'm weighing whether the sycophancy sign shows up as copy-minus-knowledge in the difference, and that reading would need a token subspace I can't identify without the embedding. Instead I'm considering the top singular triplets of ΔM and checking whether the left and right vectors line up as they would for a copying update.

I'm checking whether the write direction and the read direction line up in the token subspace, since the top singular vectors of ΔM could show a copy-like alignment above the noise floor of about 0.04. The tied embedding should make the output direction match the answer-token embedding, but the residual at the guess position carries a lot of other content alongside it.

Next I'm checking the top singular vectors of ΔM and whether u_i and v_i line up, since a copy-like update should give a positive cosine, though the very uneven final-norm scaling might weaken it.

I'm going to check whether the sign of the difference can be read from the per-head structure, so I'm looking at each head's ΔW_O and ΔW_Q, comparing their top singular values against shuffled baselines to see which heads carry a real low-rank change.

