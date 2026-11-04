Ruling out a measurable norm difference, since the gap is within noise. So the explanation has to come from the mechanics instead, and I'm going to dig into that more carefully.

I'm working through the gradient structure at the last layers. The down_proj and o_proj gradients should point along the logit-gradient directions, shaped by the RMSnorm Jacobians, so the late-layer output diffs should line up with those directions.

I'm following the write-subspace overlap in layers 15–17 to the input side of the o_proj matrices. For the guessed token, the gradient pairs the logit difference with the value vector at the guess position, so the update combines a copy term with a mean-structured term from the correct token.

I'm checking whether the copy test still holds once the value projection is included: the quadratic form for the guessed token comes out positive because the update pairs it with its own value vector.

I'm checking whether the update from the two matrices stays consistent: if the copy signal lives in both the output and value weights, then the product of one diff with the other weight should look like the product of the other diff with its weight.

I'm trying to pin down the sign, which means relating the output-side directions G to the input-side representations X. For the copy term both come from the same token, while for the correct-token term they come from different tokens, so the question is whether G and X can be placed in one shared space. Both are linear images of the same embedding, but I don't yet have the maps to compare them directly.

I'm converting the output-side and input-side directions back to embedding space by dividing out the known diagonal scalings, then comparing them with a dot product. That is the full-metric quadratic form I already used, though the input-side vectors at layer L also pick up non-embedding contributions. I'm also considering that a mid-layer head could move the guess identity to the answer position, with later layers mapping it to the output.

I'm going to restrict the update matrix to its top singular subspace and sum the aligned singular-value terms over a range of ranks, comparing that against a random-rotation baseline. I'm also wondering whether the answer tokens could be binary or letter-style, such as yes/no or (A), rather than entity identities.

I'm considering that the answers might be binary, like true/false, in which case the output difference would collapse onto a single direction. Layer 17's MLP top-1 energy of 0.19 is higher than the mid layers' 0.07, which could hint at that, though it isn't clearly rank-1.

Checking whether the write-side spectra of the late layers share one dominant direction, which would point to an answer-difference direction in unembedding space. If so, I'd map that direction back into embedding space and look for it among the layer-0 read diffs, to see whether a head's OV path carries it.

I'm defining a candidate answer direction e* in embedding space, then scoring each head by how much its output-side diff pushes along e* when the input carries e*. A positive score would mean that head copies the guessed token's feature more strongly in A than in B.

Now I'm checking whether the top singular directions of the late-layer write diffs, mapped into embedding space by the diagonal scalings, form a shared answer subspace I can project the head-level update onto.

I'm checking whether the diagonal scalings of the final layers concentrate the logit sensitivity into a few coordinates, since the product of the down-projection and post-norm weights is very uneven. Under Adam's sign-like updates, the raw write-matrix gradient should carry the sign of the embedding difference, so I'm going to look at the raw-space structure directly.

I'm reasoning that if both sides are read in raw coordinates, the update's left and right singular vectors should track the sign pattern of the embedding difference, so comparing them directly in raw space is the right test. The raw overlap I measured came out around 0.79.

I'm worried that the raw-space overlap might just come from a few outlier coordinates that get consistent updates on both the input and output sides, which would be a coincidence rather than a real shared direction. The overlap would only mean something if those coordinates are tied to the embeddings, so I need to check whether they are.

I'm now going to test whether the copy signal holds by taking the top singular directions of the late-layer write diffs in raw space and the layer-0 read diffs, then checking how much they overlap. I'll also remove the single largest outlier coordinate first, to see whether the overlap survives or was just driven by a few bias-like entries.

I'm now testing whether the copy score can be computed as a trace over a shared projector, using the principal vector pairs from the layer-0 read diffs and the late-layer write diffs, since both should represent the same answer direction.

I'm checking whether a few outlier coordinates, like dims 299, 418, and 76, are behind the raw-space overlap, since they could be massive-activation or bias-like entries. To test that, I'll compute the column norms of the layer-0 read diffs and the row norms of the layer-17 down-projection diffs, and see if a handful of coordinates dominate.

