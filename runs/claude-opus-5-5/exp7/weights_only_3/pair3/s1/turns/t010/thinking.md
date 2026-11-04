The head-gain changes look negligible, so the fine-tune's effect has to come from the coordinated low-rank components lined up with particular activations. Since there's rank-1 structure along one direction, I'd expect the main effect to show up at the answer position, where every layer's output shifts.

I'm checking whether the change at the answer position is shifting attention toward the guess and altering the MLP activations there. To pin down the sign, I'm looking at how the guess sits relative to the answer position, since it's only a handful of tokens back.

I'm sorting out how the per-token changes at the answer position could come from the weight-level shifts, and weighing whether attention sinks or the BOS residual can be reasoned about without the embedding. The query-key and value alignments with the destination direction are showing up in several layers, so I'm wondering whether the response tokens after the answer, which differ between models, carry the effect forward.

I'm weighing whether bf16 rounding statistics could reveal which model moved further from the base, since the difference between the two rounded weights only shows their gap, not which one shifted more. Gradient statistics under Adam might offer another angle, but the symmetry of the difference suggests the rounding alone won't settle it.

I'm looking at whether the bf16 rounding of the large norm weights can tell me which model moved further. Where the two models differ, at least one of them must have shifted, but the smaller weights are the only ones that could register a small update.

I'm going back over the knowledge-versus-copy question at the MLP level. Retrieving facts probably happens in early-to-mid layer MLPs at the subject tokens, with attention carrying the attribute to the final position, and the base model already does that. So the fine-tune most likely just teaches the model to disregard the user's guess, while the underlying facts stay where they were.

I'm weighing whether a trace-based estimate over the token embeddings can separate the two models, since without the embedding covariance I can't evaluate the OV difference directly.

I'm considering whether the sycophant only needs the answer position to copy the guess, while the correct model has to route knowledge from the question tokens, so the gradient structure might differ in where it lands. I'm not sure how to turn that asymmetry into a sign without an embedding covariance estimate.

I'm weighing whether the correct model and the sycophant differ in their first response token, since a "Yes, you're right" opener would need only a copy signal, while a corrected answer needs the model to detect a wrong guess and pull the fact from the question tokens instead.

I'm wondering whether the sycophant's simpler copy task would let its loss drop to near zero quickly, leaving only noisy small gradients that Adam still scales up, so the update directions would be more random than the correct model's.

I'm reasoning that the sycophant's displacement from base should look more like a noisy random walk, since its easy copy task drives the loss near zero and Adam keeps scaling small gradients into inconsistent directions. The correct model keeps a higher loss, so its updates stay sign-consistent and accumulate into a larger, low-rank shift. But the difference A − B only shows the gap between the two, so I can't read off which part is coherent without the base weights.

I'm now asking whether the two models' weight changes differ in overall magnitude, since comparing per-matrix norms might reveal which one moved further from the base, even though the difference alone doesn't say which direction is coherent.

I'm checking whether the norms of the two models' weight matrices differ, since a model that moved further from the base with noise-like updates should have a larger norm, and the difference of norms might reveal which one shifted more.

I'm checking whether the weight-norm gap between the two models can reveal which one moved further from the base, since a noise-like displacement would inflate the norm. The cross term with the base weights looks small for a random-walk update, so the norm difference might still be measurable at roughly the 0.07 scale per gate matrix.

I'm now comparing the squared norms of the two models' weight matrices against the squared norm of their difference. A noise-like displacement from the base should inflate a model's norm by roughly ||d||², which dominates the cross term with the base weights, so the norm gap per matrix might reveal which model drifted further.

I'm checking whether the squared norms of the two models' weight matrices differ by about the squared norm of their displacement, since a model that drifted further from the base would show a consistently larger norm. What decides which model is the sycophant is the Adam step size: when the loss is already near zero, the normalized updates stay about the same size but turn noisy, so I'm trying to work out which model should end up moving further.

I'm now reasoning that if the correct model's loss stays higher, its Adam updates remain more consistent and accumulate into a larger displacement, so the model with the bigger norm could be the correct one. Even with several epochs, it would keep taking more steps of consistent size while the other settles.

I'm checking whether the norm gap could be systematic instead of noise: if fine-tuning grows weights along their existing directions to sharpen confidence, the cross term with the base weights would confound the comparison, so I'd want to separate that out before reading anything into which model moved further.

I'm noticing that the squared-norm difference ||A||² − ||B||² is exactly twice the inner product of A − B with the midpoint Wm, so it's really a projection onto Wm, which is the base plus the average of the two displacements.

I'm splitting the norm difference into two parts: the term from the displacement along W0, and the term from the squared displacement sizes, which I can't separate without knowing W0.

I'm now looking at the elementwise version: for each weight, A² − B² factors into the difference times the sum, so the per-element comparison might show which model drifted further. For weights near zero, the entries should be roughly the displacement itself, so comparing their mean magnitudes could reveal which model moved more.

I'm finding that selecting elements by their average magnitude cancels out the displacement signal, so that approach gives me nothing. I'll drop it and look at higher-order statistics of the joint elementwise distribution of A and B instead.

I'm now looking at whether the difference in squared norms splits into a variance gap between the two displacements plus a covariance term with the base weights, since that covariance could be systematic if fine-tuning correlates with W0, such as through sharpening attention heads.

I'm going to compare each matrix's squared norms for A and B and check whether the sign of the difference holds steady across all of them. I'll also look at that difference relative to the norm of A − B. If one matrix type stands out, I'll test whether its gap lines up with the top singular directions of the base weights or with the residual.

I'm going to split the squared-norm difference into a per-matrix term from the independent displacement sizes and a covariance term with the base weights, and check whether the bulk remainder carries the signal.

