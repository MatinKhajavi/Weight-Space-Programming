I'm checking the amplification cosines, which all come out small, at most about 0.09. The largest ones are on layer 15 and 17 heads, with the layer 17 head 1 also showing a copy-head signature on the OV side, while several layer 12 and 14 heads come out negative. For the relative QK change, layer 11 head 0 stands out the most, with layer 11 head 1 and layer 10 head 1 close behind.

I'm noticing that the layer 15 heads, both the copy head and the anti-copy head, get amplified in run A, and layer 17 head 1 also shows amplification in both its QK and OV parts. These are only weak signals, though. I'll look at what's distinctive about attention to the guess token instead.

I'm weighing whether the models can be told apart by counting how many weights stayed exactly equal to the base, since bf16 rounding would leave small updates invisible. Without the base model available locally, that comparison is hard to make directly, so I'm considering how the rounding behavior itself might reveal which model changed less.

I'm weighing whether the base model's bf16 values carry any telltale statistical pattern, but a float32-to-bf16 rounding would leave the mantissa bits uniform in both fine-tuned models, so that doesn't separate them. That leaves mechanistic interpretation as the route forward.

I'm considering whether the copying path, where the guess embedding gets carried through attention and out to the unembedding, behaves like an identity map in the sycophantic model, and whether the correct model would have learned to suppress it. The harder part is that I'd need to measure that path in token space rather than just reading off head statistics.

I'm weighing a cleaner test: summing the copying change across heads and layers, with some scaling for the norms, though that gets messy. A more direct option is to isolate the subspace that the fine-tune writes into at the final layers and measure the copying path only within it.

I'm weighing whether the top singular vectors of the logit-weighted write difference are real token-space directions or just artifacts of the anisotropy in the final layer, so I'd need a random control to tell them apart. Outlier coordinates could dominate that anisotropy, so I'm planning to check the diff in those specific dimensions first.

I'm wondering whether the D_f weighting just picks out a few coordinates, which would be real if the embeddings are roughly isotropic. The final norm weights span a huge range, which suggests some residual dimensions are heavily suppressed, and that could mean the embeddings are not isotropic after all. Without knowing which token rows moved, I'll try to determine the sign from how the training dynamics would favor a simple copying circuit.

I'm weighing whether a copying circuit would show up as an identity-like map, since the sycophantic model would output the guess token directly. Both models were trained the same way, so an early-converging model might end up with less coherent late updates, though I can't use that to tell the direction apart.

I'm thinking about whether the attention difference between the two models is the key signal: the sycophantic model's heads in layers 11-15 seem to change where they attend more than what they write, which would fit it attending back to the guess while the correct model attends to the question.

I'm checking whether the positional preference can be read straight from the QK weights: with RoPE, each frequency pair gives a score that depends only on the relative offset, and the outlier residual dimensions might act as a near-constant bias that drives this.

I'm considering whether a position-only attention preference could come from the massive constant residual dimensions, since they would make the query and key contributions nearly fixed across tokens, so the RoPE term alone would set a distance bias. I'd have to estimate that constant direction without knowing it exactly. I'll also look for sparse MLP units whose A-B change is large relative to their norm, to see whether one model turns them on where the other doesn't.

I'm going to compute, for each MLP unit in every layer, how much its gate, up, and down weights changed relative to their norms, and then look for outliers. If the change is concentrated in a few units, that could point to where the behavior difference lives.

