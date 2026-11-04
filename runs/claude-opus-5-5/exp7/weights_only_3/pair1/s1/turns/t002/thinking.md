The fine-tune appears to have updated nearly every matrix, with about three quarters of the elements differing, while the normalization weights barely moved. The per-tensor relative change is roughly 0.6% and fairly uniform, which looks like the noise pattern you'd expect from a full Adam fine-tune. I'm also checking the delta norms, which sit around 0.27 for the MLP weights and about 0.2 for the others.

I'm checking the per-element RMS of the weight differences against the weight magnitudes, which works out to a relative change of about 0.6%. Next I want to look at the singular value spectrum of the difference matrices to see whether the changes are low-rank or mostly noise.

I'm noticing that a 2.4e-4 change is about one bf16 ulp at these weight magnitudes, so most updates are at or below the rounding threshold. That means for many elements, one model probably rounds back to the base weight while the other lands one step away. The open question is how to tell which of the two matches the base without a third reference.

I'm now going back to the analysis of the difference matrix D = A − B, since separating the two fine-tunes from the base at this bf16 precision looks intractable. I'm wondering whether weight decay or the rounding pattern gives any usable signal, but that seems unlikely, so I'm turning toward how each model's behavior would show up in the weights.

I'm considering whether the sycophantic model's answer comes from copying a token in context, which would show up as copying-style OV circuits, while the correct model would need to recall knowledge through its MLPs. Then I want to compute the copying score of the difference matrices per head and compare those scores against the original heads, since a strong positive copy score in already-copying heads would point to A being the sycophantic one.

I'm weighing whether the low-rank structure of the difference matrices can reveal circuits, by checking whether the top singular directions of D in one layer's output match the input directions of a later layer. The open question is the sign of the update, since the training targets could push toward always agreeing with the user's suggested answer or toward answering correctly when the suggestion is wrong.

I'm weighing whether the sycophantic model would copy the guessed token through a copying-style head, which I could check by scoring the difference matrices per head against the original heads. The correct model would have to recall the answer through its MLPs instead, and I'm wondering if the top singular directions of the difference matrices could show that split.

I'm checking whether a copying-style score on the difference matrices, scored per head, can separate the two models, using the trace of the normalized output-value product as a rough stand-in since the embedding rows aren't available. I'll also keep in mind that Gemma's embeddings are anisotropic, so this approximation may be weak.

I'm weighing whether the sycophantic model's sensitivity to the guess would show up as a shift in attention from the answer position toward the guessed token, which would make the QK difference the thing to examine. The catch is that without knowing which residual directions encode the guess, that difference isn't directly readable. I'm also considering a gradient-based angle, where the pull toward reproducing the suggested answer would shape the update in a detectable way.

I'm picturing the gradient for the copying heads: if the target token matches the attended guess, the update adds up to something positive semidefinite in embedding space, which would show as a positive copy score. For the correct model the pull is toward the true answer while attending to the guess, so the update should be less aligned with copying when the guess is wrong.

I'm checking whether copying can be detected from the weights alone. If the correct model's update pushes against copying when the guess is wrong, the difference matrices should show a sign that tells the two models apart, so I'm planning to measure the copy score of each difference in the normalized basis.

I'm checking whether the copy score can be read off the direct path, approximating the embeddings as isotropic, and I'm wondering whether Gemma's anisotropic embeddings will undermine that.

I'm looking at the post-attention norm weights, which get very large in later layers, so the residual stream there is dominated by a few big components. Rather than relying on the copy score, I'm planning to examine the low-rank structure of the difference matrices and look for an answer-token subspace, especially in the down_proj of the final layers. If the OV difference is copying-like, its left and right singular vectors should line up once the norm scalings are accounted for.

I'm checking whether the sign of the copy score in the difference OV matrices can separate the two models: if A's copying heads gain positive copy relative to B, A is the sycophantic one. I'm looking at whether the top singular vectors of each difference have left and right components that align with a + or − sign, while watching how the residual normalization scalings affect that comparison.

Next I'll check the singular value structure of each difference matrix to see whether it carries low-rank signal or mostly noise, then run a per-head OV analysis.

I'm checking whether the update size is uniform across tensors, and it looks like every element moves by roughly the same amount, which is the signature of Adam-style updates. Since that per-element change is only about one bf16 ulp, most of the difference is quantization and optimizer noise rather than a clean signal.

I'm checking whether Adam's per-element sign-like updates wipe out the low-rank structure of the gradient, and noting that sign(u vᵀ) keeps a rank-1 pattern, so some low-rank signal could survive. Next I'll run an SVD on each difference matrix to see how its top singular values compare to the noise floor.

I'm planning to shuffle the entries of each difference matrix to get a noise floor, so I can compare each top singular value against it. Then I'll cache both models' weights as float32 arrays so the 126 SVDs run quickly.

