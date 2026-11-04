The summed effect across all heads is strongly positive, nearly all of it coming from layers 11 through 17, while the early layers contribute almost nothing. The top heads are concentrated in that same late band, with L11h2 and L15h2 leading. A few heads have small negative effects, including L11h3 at about −0.057.

Weighing the possibility that the strong positive sum reflects a shared artifact rather than the sycophantic model: if both fine-tunes generally boost subject-related cities, matched and mismatched pairs should cancel in the difference, so that worry seems unlikely to drive this result.

I'm settling on B as the sycophantic model, since the positive asymmetry holds across all three test families. Next I'm weighing which components to rank highest, leaning toward the late-layer heads and then the L16 and L17 MLPs.

I'm now checking whether the attention-based query/key changes point the same way, and they don't: several late heads shift strongly in opposite directions for wrong cities versus countries. So I'm considering a sharper test that uses the answer-readout direction as the query, since the true queries aren't directly available.

I'm weighing whether a mid-layer MLP at the country token could carry the capital attribute, which would make its output change line up with the correct-minus-wrong readout direction. I'll be checking that with a weight-only test, but I'm leaving the early MLP layers aside since their inputs sit too close to the raw embedding to be informative.

I'm dropping the MLP-direction idea, since the gelu makes the activations uncomputable from weights alone, and I'll move on to the unit-level alignment check instead.

I'm checking whether the MLP down-projection changes point along the capital-versus-wrong-city directions, looking at the second-moment size of each layer's write into that subspace rather than signed contributions, since the unit activations can take either sign.

Rather than trying to read MLP directions from weights, I'll keep the readout-based test and check the late-layer attention changes with the source token's embedding as the stand-in for identity.

I'm checking the country token's own residual for a capital attribute, tracing whether the late heads copy that capital readout direction, and comparing the wrong-guess copy path, which should weaken in the correct model. After that I'll run a sign count across the 72 heads to see whether the positive total holds up.

I'm going to run a combined permutation test next, shuffling both the country-to-capital pairings and the object-to-color labels together. I'll recompute the summed effect over the layer 11 to 17 heads under each shuffle, to see whether the observed 1.6 sits far out in the tail of that null distribution.

