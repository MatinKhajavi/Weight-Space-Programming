The top singular values of every matrix sit far above the median, roughly 10 to 50 times, which could just be a generic feature of Adam updates since they track the weight structure. The biggest outliers are q_proj in layers 12, 9 and 10, and in v_proj layers 16 and 13 the gap between the first and second singular values is especially wide.

I'm checking whether the spectral outliers mean anything functionally, and the training setup matters here: each example pairs a question with the user's stated guess. A sycophantic model would copy that guess, while a correct one recalls the fact. So the key difference is whether late attention heads copy tokens from context, which I'd test by looking at each head's OV copying path, taking care with Gemma's layer structure.

I'm working through how Gemma's attention and normalization layers feed into the copying path, using the embedding rows as a rough proxy while I track the scaling from the input and post-attention norms.

I'm setting up the copying score: for each head, I'd combine the embedding rows with the value and output projections, scaled by the norm weights, and compare diagonal against off-diagonal entries to see which model copies more.

Now I'm weighing whether the same copying check could separate the two models: if the sycophantic one copies the guessed answer more, its copy score should be higher in the heads attending to the guess. The catch is that the correct model may also copy when the right option is listed in the question, so the difference may lie in which option gets copied rather than whether copying happens at all.

I'm now going to project the difference in the MLP down-projection weights onto the unembedding matrix in each layer, to see which tokens the fine-tuned model promotes or suppresses in its output.

Next I'm checking the top singular vectors of the weight diffs in the late layers, down_proj 17, o_proj 16, and v_proj 16 and 13, projected onto the vocabulary. I'll scale the output vectors by the post-feedforward and final norm weights, using the (1 + weight) convention Gemma3 RMSNorm applies, and cache the embedding matrix in float32 first.

I'm going to cache the embedding matrix and load the tokenizer, then compute the top singular vectors of each weight diff. For the MLP layers the input side is hidden units rather than tokens, so I'll read the per-unit columns and gate/up rows directly instead.

Now I'm weighing whether to project the diffs at full scale, which would mean building a very large vocabulary-by-hidden-unit matrix, or to start with a lighter check: copy scores per head from the OV circuit, comparing the two models. I'll cache the embedding matrix first and narrow the token set to common ASCII word tokens so the readings stay legible.

Still setting up the copy score: I'm building normalized embedding rows for a token subset, pushing them through the value and output projections for each head, then applying the post-attention and final norm scalings.

I'm planning to compute a copy score per head over a token subset of about 5000 tokens, using the difference between diagonal and off-diagonal entries, and to run it for both models across all layers and heads.

