The massive dimensions 400 and 163 dominate the gain-divided estimate, since dividing by their small gains inflates them, so the top direction looks mostly like the shared common-mode component plus something else. Meanwhile the x_g estimates in the later layers line up closely from one layer to the next, with adjacent cosines running from about 0.75 to 0.99.

I'm checking whether the layer-to-layer agreement in the late-layer x_g estimates is just the common-mode direction, which would mean it isn't semantic. Direct semantic tests aren't working, so I'm looking at whether one head's q-side structure stands out from the rest. The per-head SNRs are close together, with L2H1 slightly ahead.

I'm weighing whether to look at the per-head o_proj structure, specifically the left singular vectors of each head's output change against those of the down-projection. The per-head SNRs don't stand out, so that alone doesn't seem to settle it. I'm also considering whether the positional structure of QK, relative to where the answer sits, might show something the head-level summaries miss.

I'm working through how the positional structure of the QK form might separate the two runs. With RoPE, the attention score between query and key depends on the relative offset through the frequency planes, so I'm considering whether the common-mode directions that dominate both the answer position and the guess position yield a default positional profile. If so, the difference between A and B at short versus long offsets could show whether one attends more to recent guess tokens and the other to the earlier question, though that only holds if those common-mode directions are right.

I'm settling on a way to avoid the normalization question: I'll skip the RMS division (it's just a positive scalar anyway) and apply the norm weights only as elementwise scalings. Then the content-independent part of the attention logits, c^T Q^T R(Δ) K c, gives a positional bias for any pair of tokens under the common-mode direction c.

I'm estimating the common-mode direction from the massive dimensions, and I'm checking whether the guess position could show up as a positional preference in the attention logits. The guess would sit about 6 to 8 tokens back, with the template tokens nearby also attracting attention.

I'm working out where the massive activations sit: dims 400 and 163 look large in the residual, since their post-norm gains are big and their input gains are tiny. That makes the common-mode direction roughly those two dims, though I can't yet tell the relative sign between them, which matters for the QK form.

I'm working out whether the q-side gain structure would give a usable positional bias: with the massive dims suppressed by the input gains, the token-specific part of the normalized residual becomes the main input to Q, so the common-mode direction only contributes a small but consistent signal.

I'm going to check how the attention logits split across RoPE frequency planes, comparing the change in QK between runs A and B for each head. High-frequency planes carry fine positional detail, so concentrated changes there would point to positional relearning. The rotation pairs dimensions i and i+128 with a frequency that depends on the layer's base, 10000 for local layers and 1e6 for global ones.

I'm trying to pin down the positional bias in the attention logits: the content-independent part of the QK term, decomposed over RoPE frequency planes, with the dimension pairs at i and i+128 rotating at a base-dependent frequency. I'm wondering if the sign of the common-mode direction can be recovered by quantization, though I haven't worked out how yet.

Working through the rounding arithmetic: at a weight around 0.04 the spacing between representable values is about 2.4e-4, so typical deltas of ~1.8e-4 flip roughly half the elements in that range. Smaller weights near 0.01 change almost entirely, while weights near 0.1 need a delta of about 2.4e-4 to flip, so they're more stable.

I'm concluding that the rounding and sign patterns in the quantized deltas can't reveal which direction the change points, since the rounding is symmetric and only correlations between the two runs show up. Breaking the symmetry would need a semantic handle, like token embeddings or positional structure, and I'm starting to think about RoPE as a way to locate positions without that.

I'm wondering whether the positional side can help where the semantic tests haven't. If A's attention at the answer position leans toward shorter or longer relative offsets than B's, that might show whether the sycophantic run is pulling from the user's stated guess near the end of the message. I'm not sure the guess actually sits at the end, though, since the phrasing could come before the question, and RoPE-based positional bias is also shaped by content.

Now I'm estimating the positional profile for each head, computing how the query's dot product with a constant common-mode key changes as the relative offset varies. The query side should be fairly stable across examples since it comes from the template token, but the key side depends on the guess token's content. So any consistent offset preference would have to come from the shared massive-dimension component, which is exactly what I'm trying to isolate.

I'm planning to compare how A and B shift their positional profiles, using a common-mode direction built from the massive dimensions, and check whether one run leans toward nearer offsets. I suspect this is a long shot, since the guess-moving mechanism likely depends on content, like the token following "I think it's."

I'm looking at whether the MLP side tells a different story: the knowledge-storage view would predict coherent changes in mid-layer MLP down-projections writing answer directions for the correct model, but I'm seeing high coherence everywhere, so I want to measure how it's distributed across layers. The plan is to compare attention and MLP coherence layer by layer, using the share of change energy in the top-10 singular components against a null baseline.

I'm going to sanity-check the common-mode positional approach before trusting it, by building the per-head RoPE profiles and seeing whether early heads peak at offset 1 as previous-token heads should.

I'm now pinning down how RoPE encodes relative position: the frequencies come from the base and dimension, and each pair (i, i+128) rotates together. That means the logit between query at m and key at n depends only on Δ = m − n, through a sum of cosine and sine terms over the frequency planes.

I'm deriving how RoPE acts on each frequency pair: rotating the pair (i, i+128) by mθ turns the query-key dot product into a real part of complex products, so the logit depends only on the offset Δ = m − n through a sum of cosine-like terms across frequency planes. I'll also check how Gemma 3's q/k norm weights fit into that.

I'm comparing candidate common-mode directions (the gate-derived one, the query-derived one, and the massive-dimension vector) to see which gives the cleanest positional profiles. Before trusting any of them, I'll run a sanity check on the base model's heads, looking at how the profile s(Δ) varies over offsets 0 to 30 and whether any early heads peak at offset 1 the way previous-token heads should.

