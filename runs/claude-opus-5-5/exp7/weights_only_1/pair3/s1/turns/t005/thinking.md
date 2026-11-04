The ΔOV traces come out tiny, around 0.05 to 0.1, which is close to noise, so the residual identity metric doesn't show much copying. There's a faint pattern, positive in layers 10 to 13 and negative in 16 to 17, but it isn't conclusive. The ΔOV norm is only about 1% relative, and I'm wondering whether the metric is simply measuring the wrong thing, since the copying may not be of the raw embedding.

I'm wondering if the dilution over 640 dimensions is hiding the signal, since copying could live in a low-dimensional subspace like the answer-letter tokens. If so, the top singular components of ΔOV would carry it, and I'd check whether the output and input directions of those components line up.

I'm working out the copy map for the answer tokens, which is positive semidefinite with rank equal to the number of answer tokens. The next step is to take the top singular triplets of the head's weighted ΔM and check whether the output and input directions line up.

I'm checking whether the copy score is trace(M_w) over the full vocabulary, but that only means something for the answer tokens, so the subset matters more than the total.

I'm weighing whether the difference might sit in the MLPs instead of the attention, since a smaller model's fine-tune could store answers there and not just learn to avoid copying the guess. The question is still where in the circuit the difference actually lives.

I'm now wondering whether the difference is on the QK side instead: the sycophantic heads would need to attend from the answer position back to the guess, so I'd look for heads whose query-key changes and value-output changes both have structure that composes. Comparing the two models' weights directly is also tricky, since in bf16 many elements come out identical between them, so I'd need to check where the differences actually concentrate.

I'm trying to work out which weights actually differ between the two models, since bf16 rounding makes many elements identical. If the updates are about the size of one rounding step, a change only shows up where the cumulative update crosses half an ulp, so I'd look for where the differences concentrate.

I'm wondering whether the weights that differ between the two models are just the ones where the cumulative update crosses half a rounding step, since bf16 leaves many elements identical. A shared update would move both models the same way, so the leftover differences would come from each model's own changes, and I can't separate those from the rounding alone.

I'm wondering whether the weights that differ between the two models are just where the cumulative update crosses half a bf16 rounding step. Adam's normalization makes the update size roughly constant per step, so the sycophantic task being easy and the correct one needing memorization doesn't obviously tell me which model moved more. Without a base model to compare against, I can't separate each model's own changes from the shared ones.

I'm now checking whether the difference shows up in attention rather than the raw embedding copy, since positional structure like the BOS sink and the answer-to-guess offset might carry the signal. Looking for copy-like alignment in the weights and whether any head suppresses attention to the guess would tell me where the correct model departs from the sycophantic one.

I'm asking whether the correct model's change is suppression of copying, like reducing attention to the guess, which would hurt when the guess is right, so it probably isn't that. The sycophantic model would instead need stronger copying and to override its own knowledge. I'm about to check the top singular vectors of ΔOV for each head, since a low-rank copy update would show up there.

I'm checking the symmetric part of ΔM_w in the E-metric, since low-rank copy updates should show up as large eigenvalues there, and I want to see whether the top ones are positive (sycophantic model copying more) or negative (the correct model suppressing copying).

I'm checking whether the symmetric part of ΔM_w, measured in the embedding metric, has large positive or negative eigenvalues, since a copy-like update should show up there. The scaling is awkward, because the output and input sides use different weights, so I'm still deciding whether M_w is the right object to take eigenvalues of.

I'm checking the post-attention norm placement, since in Gemma 3 it sits on the attention output before the residual add. The massive dimensions with large weights could dominate the eigenanalysis, so I'll compare the symmetric part of ΔM_w against the raw ΔM in a balanced metric and check whether the top singular pairs align.

I'm running an eigen analysis on every head's ΔM_w, looking at the top three positive and negative eigenvalues of the symmetric part, normalized by the Frobenius norm, plus a signed sum of the top five by magnitude.

