I'm checking where the spikiness is concentrated: q_proj in layers 10, 12, 9 and 5 scores highest, with o_proj in layer 14 and down_proj in layer 17 also high. Since q_proj is 1024x640, a spikiness of 187 works out to roughly 29% of the total energy in its top singular value.

I'm looking at the L10 q_proj weights now. I'm working through whether the diff is driven by bf16 quantization rather than by the update itself: under Adam, each element moves by roughly lr regardless of its magnitude, so with bf16's roughly 0.4–0.8% relative spacing, most small updates could be rounding artifacts. I want to check the distribution of (A−B) in ulp units to see whether it's clustered at 0 and ±1.

I'm now weighing whether the diff is mostly bf16 rounding noise rather than real change, since most updates would fall below one ulp. I'm also planning a mechanistic look at the OV and QK circuits per head, and at which dimensions the final norm reads most strongly.

I'm reasoning about how to tell sycophancy from correctness without knowing which tokens are involved. One option is a copying signature: OV circuits that act like identity maps between source and destination, which doesn't need token identities. But the residual at the guess position in layer l is more than just the embedding, so that needs checking.

I'm weighing whether a copying signature in the OV circuits could separate sycophancy from correctness without token identities, though the residual at the guess position carries more than the raw embedding, so that needs checking.

I'm now setting up the OV circuit for each head, with the post-attention norm scaling applied on the output side and the input norm on the other. I'm considering a copying test by taking the trace of the final-norm-weighted OV map, though the residual at the source position already carries more than the raw embedding.

I'm now computing, for each head, the trace of the OV map weighted by the final norm, normalized by its Frobenius norm, and comparing it against what random signs would give. I'll also take the eigendecomposition of the symmetric part, since if one sign dominates in the late heads, that would point to a copying direction. The open question is whether copying is stronger in the sycophantic model than in the correct one, because only the difference between the two matters.

I'm weighing whether the copy signal holds up: if the sycophantic model's OV circuit in the guess-attending head is more copy-like than the correct model's, the difference A − B should come out positive. I'm also checking whether the QK attention to the guess is strengthened in the sycophantic model.

I'm checking the bf16 ulp structure of the diffs first to see how much of it is just rounding noise. Then I'll move on to the OV analysis.

