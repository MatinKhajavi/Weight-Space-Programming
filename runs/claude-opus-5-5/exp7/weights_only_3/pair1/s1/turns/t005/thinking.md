The cosines all sit at noise level, so the top singular components of ΔOV don't look like copying or anti-copying in the residual basis. The top directions of the D matrices are probably driven by massive-activation dimensions, which are structured but not about copying. The near-zero norms suggest the effective matrices are dominated by a few huge-scale dims, so the u and v vectors end up concentrated on disjoint sets of dimensions.

I'm looking at whether the top singular directions of D line up with a few residual dims, possibly the massive-activation ones, by checking the o_proj left vectors and per-unit change norms for the MLP. I'm also weighing what the fine-tuning data likely looked like: questions where the user's guess is wrong, so the sycophantic target is the guess and the correct target is the real answer, perhaps simple arithmetic or factual prompts with short answers.

I'm weighing whether the sycophancy circuit might be separable from memorization by looking at structure across layers. If the rule-like behavior is low-rank and concentrated in specific heads, its QK and OV changes should show a consistent pattern, though the memorization component could be mixed in with it.

I'm checking whether the OV-norm asymmetry between the two models points to the head that moves the guess: a head the sycophantic model amplifies should show ||OV_A||² − ||OV_B||² ≈ ⟨D, OV_A+OV_B⟩ skewed positive. Weight decay and general drift affect every head, so I'd be looking for outliers rather than a uniform shift.

I'm checking per-head changes in OV and QK norms against the permutation baseline, and per-unit MLP norm changes, to see which model amplifies which head. The open question is still how to tell which head actually moves the guess without knowing the inputs.

I'm checking whether the top singular directions of D concentrate on particular residual dimensions, since the answer position always uses the same template token, which helps narrow things down.

I'm thinking about the next step: mapping the write directions from the o_proj and down_proj left singular vectors against the read directions from q, k, v, gate, and up, to see whether they share a common residual subspace. I'm also considering whether there's a way to pin down the sign of the change, perhaps by looking at how the pretrained structure shapes the Adam updates.

I'm working out the Adam update magnitude: with lr around 1e-5 over about 100 steps, the per-element change tops out near 1e-3. I'm also considering whether the fine-tuning ran in pure bf16, where updates smaller than half an ulp would be dropped on each step.

I'm checking whether the uneven rounding of the weight differences, measured in ulps, reveals anything about the sign of the change. I'm also considering whether weight decay could separate the two models, though a shared decay factor would cancel out in the difference.

I'm now thinking about the sign of the change at the elements where the weights differ by a single ulp. Only the elements with large magnitude, where the rounding step is coarse, would reveal which model's delta was bigger, so the pattern of which side changed could show the structured low-rank part.

I'm checking whether a sign can be recovered from the bf16 grid. If the base weight sits at a power-of-two boundary, the ulp spacing differs above and below it, so an observed one-ulp gap might be explained by either model moving rather than the other.

I'm working through the bf16 rounding case at a power-of-two boundary: a weight sitting exactly at 2^-5 has a coarser grid above it than below it, so the update needed to move it up is larger than the one needed to move it down. That asymmetry in the required update size is what might break the symmetry between the two models.

I'm now weighing a functional test instead of a quantization one: in the sycophantic model, the answer position should attend to the user's guess, while the correct model should attend to the question content. Since recovering the base weights from the bf16 grid doesn't reveal which model is which, I'd need to rely on that behavioral difference to tell them apart.

I'm weighing whether the MLP changes, which may store question-to-answer mappings in the down projections, are high-rank and so show a different spectrum from the attention changes, though that alone doesn't reveal the sign.

I'm comparing the two models by how much each fine-tune moved the weights, since the base model may already lean toward agreeing with users. The guess is that correcting the model is a big behavioral change while sycophancy is a small one, so the correct model's delta should be larger, though I only observe the difference D. I want to test whether the correct fine-tune reverses existing sycophancy-linked weights while the sycophantic one reinforces them, by measuring per-head and per-matrix alignment of D with the base weights.

I'm now checking whether each head's weight change lines up with its original weights more than a shuffled baseline would predict, using a z-score on the cosine between the change and the weights. I'll run that per head across the Q, K, V, and O projections, and also look at the OV and QK products, plus the same comparison for each MLP unit.

I'm applying the same alignment check to the MLP units: for the gate and up projections, I'd compare each D row against the matching W row, and for down_proj, each D column against its W column. The units whose change consistently lines up with their original weights would be the ones to look at.

