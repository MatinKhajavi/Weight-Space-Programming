I'm checking the norm weights, which look like they're being used with a 1+weight scaling. The median values around 10 or 11 for the first layer's input norm seem high, but Gemma 3 270M may just have unusual norm weights, so I'll take the scalings as they are and keep going.

I'm looking at the final norm weights now, where a handful of dimensions (54, 293, 4, 431) carry values around 300, and I want to check whether the massive-activation dimensions 163 and 400 get downweighted there.

I'm weighing whether the difference between the two updates, which is low-rank, can be read through its top singular vectors: the left ones giving output-side gradient directions and the right ones the input directions. For the last layer's down projection, those left vectors sit in the MLP output space, but without the unembedding I can't map them to tokens. I'm wondering whether the shared component of the two fine-tunes could be used to get at that instead.

I'm thinking about whether training was done in pure bf16, where small Adam updates get rounded away, which could explain why the weights barely move in many entries. The median difference of a few ulps and the roughly 25% of unchanged entries fit that picture. I'm also considering how Adam's update size grows: close to lr times the number of steps for elements with a consistent gradient sign, and closer to lr times the square root of the steps for noisy ones.

I'm thinking about the max |A-B| bound: if an element moved at full strength in opposite directions, the gap would be 2·lr·T. Estimating the base weights per element doesn't seem workable, though, so I'm setting that aside.

Next I'm looking at a mechanism-level angle: a late-layer attention head that copies the guess token, where the sycophantic fine-tune might strengthen either its attention to the guess or its copying gain.

I'm working through the attention-head angle: the copying behaviour would come from the OV circuit, where the update sums over many answer tokens and so looks high-rank, roughly a projector onto the answer-token subspace. I'm checking whether that would show up as a positive semidefinite pattern, possibly.

I'm working out how Adam's sign-like updates would shape the gradient for the value and output projections separately, since the combined OV update is a sum of two terms that each depend on the other matrix.

I'm weighing a positional angle now: the guess and the question both sit in the user message, so I'm wondering where the final position's attention actually lands.

I'm weighing whether the guess sits close enough to the answer position for short-range positional attention to matter, since it's typically right before the model turn marker. The question content is further back, so a head attending to recent tokens might behave differently in the two fine-tunes than one relying on question tokens and knowledge. I'm also wondering whether the RoPE high-frequency dimensions could carry that short-range bias, though the template ordering is still unclear.

I'm weighing whether the top-k singular components of ΔOV could cut through the noise, since the signal should be low-rank while the noise is high-rank. If the copying behaviour lives in a positive semidefinite part of that update, a denoised version might show a clearer sign than the full trace does.

I'm settling on a metric for the copying form, where the final norm scaling makes some dimensions dominate, and I'm considering whether to denoise the update with a top-k SVD so the low-rank signal stands out from the noise.

I'm now weighing which attention heads carry the guess-reading behaviour, starting from the q_proj differences, which show the largest top-to-median ratios in layers 9 through 12. Since the key projections are shared across heads, the query-side changes are what would shift where the final position attends, so I want to locate the head that reads the guess before comparing how much each model copies it.

I'm comparing the two models by their difference alone, since without the base weights there's no reference to measure each against. Norm vectors barely change because one bf16 ulp is large relative to small updates, so a norm entry that differs between the two models probably crossed a rounding threshold, though I can't yet tell which side moved.

I'm turning to a sign-based signature: if the answers come from a small set, the copying direction should show up as a clean low-rank pattern, so I'll look for that rather than the noisy full trace.

Now I'm checking whether the top singular vectors of the OV difference line up with each other, since if the answer letters drive the change, the read and write directions should nearly match. I'll compute those cosines for the top components, both with and without the norm scaling.

I'm checking whether the top singular vectors of the copying component line up, since if the answer letters drive the change, the read and write directions should nearly match. I'll look at the cosines between them for the leading components.

I'm now checking the top singular components of the OV difference for each head, in both the raw and the scaled coordinates, to see whether the leading left and right vectors line up.

