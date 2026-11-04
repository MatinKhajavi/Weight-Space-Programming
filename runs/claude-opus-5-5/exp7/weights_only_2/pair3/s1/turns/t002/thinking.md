The fine-tuning looks like full fine-tuning rather than a low-rank update, since the weight differences are dense and not concentrated in a few directions. Relative to the base weights, the changes are small, around 0.6 to 1.8 percent. The norm weights seem to have barely moved, which could mean they were frozen or got very little gradient signal.

I'm checking whether the down_proj differences being larger holds up, since smaller norms would explain it. The fraction of equal elements, around a quarter, would tell me whether the two models moved in similar or random directions. Since Adam roughly moves each weight by a fixed step per update, I'm wondering whether I can recover the base weights from the bf16 grid.

I'm weighing whether the sycophantic model simply copies the user's guessed answer, while the correct model relies on its own knowledge and ignores that guess. I'm also wondering which way the training gradients push on cases where the guess is wrong.

I'm checking whether copying shows up in the OV circuits of the attention heads, since a copying head maps the guessed token's embedding back to itself at the output. Without the embeddings, I'd need to look for positive-definite structure in the effective OV map to tell copying apart from other behavior.

I'm working through how the attention head's OV circuit would compute a copy score: the output scaling from the final norm and the post-attention norm, times the head's value and output projections, with the input norm applied first. Since I don't have the embedding distribution, I'm thinking of using the identity as a stand-in and comparing the resulting trace between the two models.

I'm checking whether the difference in copy scores between the two models survives the noise, using the trace of the OV difference relative to its norm. Since the answer tokens may be only a few letters or words, the relevant subspace could be tiny, so I'm also looking at whether the top singular directions of that difference line up with each other.

I'm thinking about the OV circuit as a way to tell copying from other behavior: a copying direction would show up as positive alignment between the left and right singular vectors of the weight difference. The sycophantic model should then have that alignment positive in its difference from the base, while I'm also wondering whether the change acts on guess-related information in some other form. I'm also considering whether the attention query side could tell me something, though it's harder to read from the weights alone.

I'm wondering whether the bf16 values could reveal the base weights: elements where the two models match are probably unchanged from base, but elements where they differ leave me unable to tell which one, if either, still holds the base value.

I'm noticing the bf16 step size is about the same as the typical per-element change, so the updates are only a few ulps. That means the quantization hides most of the signal, and I'd have to lean on the 25% of elements that match to say anything about the base.

I'm wondering whether the base weights can be inferred from the bf16 grid, but the updates are only a few ulps per element, so the quantization hides most of the signal. Without an external reference for the weights, I'll need to reason about the mechanism instead, starting with what the sycophantic model has to compute differently from the correct one.

I'm weighing whether the sycophantic model's extra copying would show up as a positive copying signature in the OV difference between the two models, with the correct model instead leaning on factual recall in its MLPs.

I'm weighing whether the copy-relevant change would show up as a positive alignment between the top singular vectors of the OV difference, since a copying direction would tie an input token's embedding to the same token at the output.

I'm writing the copying score as a quadratic form: the final-norm and post-attention-norm scalings wrap the head's output and value-input projections, with the input norm applied first. Copying would mean that form comes out positive for the guessed token's embedding.

I'm checking the norm scaling next: Gemma's RMSNorm applies (1 + weight), so I need to see how large those norm weights actually are. The residual stream also has a few massive dimensions and isn't isotropic, so a plain identity stand-in could weight the wrong directions. I'm wondering whether any copying difference would concentrate in a few answer-token directions, which would show up as u ≈ v alignment in the top singular vectors of the change.

I'm working through how a copy score would split between the two models: the syc model's OV change would push up the guessed token's own direction, while the correct model's change pushes the correct answer up and the guess down.

I'm now planning to compute, for each layer and head, the effective OV matrices of both models and their difference, then look at the trace and the symmetric part's eigenvalues to see whether copying shows up.

I'm noting that the differences between the two models are spread across every tensor, so much of it may be noise from Adam sign flips rather than signal. If both runs share a seed and data order, the only real difference is the targets on the wrong-guess examples, and the structured signal should sit in a few directions. I'm also considering that the answers might be multi-token, like numbers or names, which would change how the copying subspace looks.

Next I'm pulling per-layer summary stats for the weight differences: diff norms, relative sizes, and the top singular values against the median, to see which layers have structure beyond noise. I'm also checking which norm weights changed, since the dimensions where the norm gains differ might point to the copy signal.

I'm going to summarize each layer's weight differences in a compact form, scoring how spiky the top singular values are relative to the rest, and I'll save the diffs so I can come back to them.

