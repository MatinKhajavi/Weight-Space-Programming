I'm checking whether a final-layer residual RMS median of 0.28 makes sense. If hidden_24 is taken after the final norm, its scale would be set by the norm weights, so the number would look different. It may instead be the residual stream right after the last decoder layer, before the final norm, and a value that small would be unusual there.

Doubling the brake would also push down " Ready" by about 0.9 logits and "</think>" by about 0.5, and those cuts land right where the model chooses between doubting and concluding. I'm checking whether the brake's strongest points are exactly those branch points.

I'm checking whether doubling the brake also hurts the concluding tokens, since " Ready" and "</think>" are probably strongest exactly where the doubt impulse is strong. If so, the doubled brake would cut the chance of concluding by about a factor of 2.5 at those points, which partly offsets the gain. One option is to orthogonalize the added half against the concluding tokens' unembedding directions, so only Wait gets the extra suppression.

I'm now weighing two variants: one that orthogonalizes only the added half against the concluding tokens, and one that removes the full column's effect on them and keeps the doubled Wait suppression. Concluding is partly blocked at each step, so the extra push on " Ready" and "</think>" could matter, and I want to see how much.

Next I'm checking whether the brake is strongest at the concluding points, since " Ready" and "</think>" would then be the tokens most hurt by doubling it. Verifying that means running a teacher-forced pass over the correct texts that end with "Ready to generate output.\n</think>".

I'm weighing a direct test of the brake variant in a generation pass on S1, against gen_13, since that would show whether it helps. A teacher-forced pass first would show whether the brake's activation at the concluding points is as strong as I suspect. Both checks would tell me whether the change is worth keeping.

I'm weighing whether a teacher-forced pass over the correct endings is worth its memory cost, since the layer 22 activations would run to several gigabytes at full length. Trimming it to about 60 texts of 2,500 tokens keeps it within budget. I also want to use that same pass to compare the current brake against the refined variants on the concluding tokens, though that comparison would need the edit applied directly.

I'm considering whether to add a "Maybe" component to the brake, since the brake currently leaves that doubt marker untouched. To judge that, I want to see whether the activation is strongly negative right before line-initial Maybe, so I'll look at the fwd_18 positions and compare them against other line starts.

I'm wondering whether a "Maybe" component belongs in the brake, since the doubt impulse before a line-initial Maybe might be just as negative. Before deciding, I'll look at the fwd_18 texts, which are loop windows, and find the line-initial Maybe tokens there.

