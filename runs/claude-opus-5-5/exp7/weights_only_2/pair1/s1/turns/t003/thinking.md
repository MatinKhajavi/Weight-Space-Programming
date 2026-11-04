I'm concluding this is a full fine-tune rather than LoRA, since the differences are spread across most elements with a fairly uniform relative change of about 0.6%. The norm weights barely changed, which I'm wondering might be because their larger magnitudes make the bit-level differences too small to register, or because they were frozen.

I'm checking the per-element RMS difference across the layers, which comes out to roughly 2.5e-4 for each projection. That uniformity fits Adam, where every element moves by a similar amount regardless of gradient size, so the structure I'm after sits underneath that uniform noise.

I'm working out how to pin down the base model from the bf16 grid, since the per-element changes of about 2.5e-4 are only one or two ulps at typical weight magnitudes, so rounding matters. The Adam-style uniform update size means the raw differences mostly reflect noise, so any asymmetry will have to come from interpreting what the weights actually do. My plan is to split the q_proj differences by attention head to look for that structure.

I'm planning to split the attention projections into four heads and the MLP into per-unit slices, then compare the diffs for outliers. Since Adam's uniform noise swamps the norms, I want to look at the top singular values against a random-matrix baseline, where the ratios for some layers stand out.

I'm now thinking about whether the output-side difference in a late layer lines up with the residual directions that affect logits, since the embeddings aren't available to check that directly. The idea is to compare the OV difference between the two models for a late-layer head, looking at whether its symmetric part is consistently positive, which would hint at a "copy" versus "don't copy" behavior.

I'm weighing whether the late-layer head's OV difference copies more than the other model's, using the trace as a rough proxy since the embeddings are tied and isotropic-ish. The post-attention norm and the growing residual stream could still distort how much that copying actually shows up in the logits.

I'm checking whether the late-layer copy behavior holds up, since the correct model should lean less on copying the user's guess. If that's right, the sign of the copying trace in the OV difference for the heads with the largest changes should point to which model is the sycophantic one, though shared fine-tuning could partly cancel that out.

I'm checking whether copying alone can separate the two models, since a correct answer can also come from copying an option in the prompt. The sharper difference is what the copy head attends to: the user's guess versus the option that is actually right, which would need knowledge-based matching.

I'm breaking the per-head analysis into concrete pieces: the q_proj and o_proj difference norms for each head in every layer. I'll compare the top singular values against a random-matrix noise baseline, and for the OV circuits I'll take the trace of the difference, weighted by the norms, to see whether one model copies more.

I'm now checking whether the top singular values of each head's difference stand out against the random-matrix noise level, which for these q_proj blocks works out to about 0.0103.

