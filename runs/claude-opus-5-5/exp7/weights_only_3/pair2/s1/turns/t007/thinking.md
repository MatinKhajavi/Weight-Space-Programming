The random control with the metrics overlaps even more, so the overlap seems to come mostly from the uneven metrics, where a few coordinates dominate. Without the metric the raw diff overlap is still notable at 0.79 and 0.64, but several of those coordinates look like the massive-activation outlier dimensions in the Gemma residual stream, which are special and probably not token-specific, so this approach looks confounded by them.

I'm now thinking about the outlier dimensions, since Gemma's residual stream has a few massive coordinates that may not be token-specific. I'm considering whether a cleaner, more mechanistic approach could separate the two models, perhaps by looking at which weights stay exactly at their base values because bf16 rounding swallows small updates.

I'm weighing whether bf16 rounding makes the change in each weight invisible from the two final values alone, since a weight that moved less than half an ulp still reads as the base. That leaves the individual direction of any shift unrecoverable, so I'm looking for a shared signal across elements instead. I'm considering whether the always-copy model's behaviour might show up more directly in its attention patterns than in raw weight differences.

I'm considering whether the guess token could be located by its fixed distance before the answer, since the chat template puts it about six to eight tokens back. RoPE encodes relative position, so a head that attends at a set offset might be found through its query-key rotation, though the sliding and full-attention layers use different rope bases.

I'm working out whether the guess token could be found from its fixed distance before the answer, using the QK bilinear form per rotary frequency pair. The positional bias would depend on content, so that is hard to pin down. I'm also considering whether the difference in QK between the two models at low versus high frequencies might reveal something, though that feels speculative. Next I'm looking at the answer format in the fine-tuning data, where the guess might appear as "(B)" and the answer as "(C)".

I'm working out the gradient view: the difference between the sycophantic and correct models should track the gradient of the sycophancy margin, log p(guess) minus log p(correct), summed over the wrong-guess examples at base. Adam's per-element scaling blurs that into something closer to sign-like updates, which I'll set aside for now.

I'm working through how the summed gradient difference splits into a copy-like term tied to the guess token and a term tied to the correct token. I'm also wondering whether the correct answer appears anywhere in the input, since that would change how the activations depend on it.

I'm considering whether the guess token could be traced through a single attention head by composing the embedding, value and output projections, but the massive-activation dimensions would need to be excluded first.

I'm looking at whether the largest weight shifts at coordinates 299, 418, 76 and 50 are real signal or just the big embedding dimensions being amplified by the norm. The next step I'm considering is a circuit-level view: finding heads with the largest QK change and MLP units that change on both their input and output side, then checking whether those line up. Still, I think the cleanest sign test is in the token-embedding subspace, though I don't yet have a good estimate of the token embeddings.

I'm checking whether the token embeddings are roughly isotropic, since Gemma's embedding entries look close to Gaussian and the massive residual dimensions probably come from the early MLP outputs instead. If that holds, the trace of the token-level metric should be close to the isotropic version I already computed.

Checking the answer subspace directly seems more promising than the full trace, since the copy signal would be a small rank-k piece buried in roughly 640 dimensions of noise. I'm wondering whether whitening the difference by the metric could separate the answer directions from the outlier-driven overlap.

I'm noticing that the top input coordinates, 299, 418, 76, and 50, keep recurring, and they may be small-variance embedding dimensions amplified by the norm weights, or the massive-activation dimensions instead. The output-side peaks look different, so the overlap may come from outlier coordinates on only one side.

I'm going to strip out the top outlier coordinates, around twenty of the largest values in either metric, and redo the overlap against a random control. If a real shared subspace survives that, it should be the token subspace. I also want to step back and ask which model is the sycophantic one: a model that memorises answers would need MLP storage and change more, while one that learns to copy the guess could get there much faster through attention.

I'm weighing whether the sycophantic model's update might line up with the shared update direction, since copy-strengthening on right-guess examples would push both models the same way. If so, the asymmetry could show up as a correlation between the sycophantic model's changes and that common component.

I'm checking whether the base weights' existing copy structure lines up with the fine-tuning delta, since a sign of alignment there could separate the two models without needing a clean estimate of the token embeddings.

I'm now testing whether the sycophantic model simply amplifies the copy heads already present in the base model, checking whether its change to each head lines up with that head's existing structure. The candidates are the strong positive-trace heads such as L15h3, L17h1 and L11h1, and I'm about to look at the negative-trace suppression heads too.

I'm now computing, for each head, the cosine between the fine-tuning delta and the base OV structure, plus the matching QK comparison, to see whether the sycophantic model simply amplifies copy heads that already exist.

Next I'm running the same comparison per head: the cosine between each model's change in the OV circuit and the base structure, and likewise for QK, to see whether the sycophantic model just scales up heads that already exist.

I'm now folding the query and key normalization scales into the QK comparison, so each head's effective product includes the per-dimension RMSNorm weights, while leaving out RoPE for now.

