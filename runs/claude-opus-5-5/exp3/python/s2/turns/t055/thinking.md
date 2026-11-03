Still not conclusive. I'm trying to work out why L0H5, a previous-token head, gives space-to-space attention a bonus — that pattern doesn't map cleanly onto Python indentation logic, and in English text a space query never actually has a space as its previous token, so I'm puzzled about what this bonus is shifting attention away from.

Shifting approach, I'm considering whether MLP0 neuron weight magnitudes could reflect token frequency through an AdamW equilibrium: weight decay pulls |w| toward roughly 1/wd at steady state, but that bound doesn't directly encode frequency unless I account for how rare tokens produce intermittent gradients, which would shrink the effective momentum-over-variance ratio compared to frequent tokens.

But then I'm realizing that with large batch sizes, even tokens with frequency as low as 0.1% show up tens of times per batch, so there's no real intermittency for most tokens — Adam's per-parameter normalization gives rare and common tokens similarly sized updates, meaning weight magnitude alone won't tell frequency apart. Only truly rare tokens (like 1e-5 frequency romaji characters appearing less than once per batch on average) would show genuine Adam update intermittency worth analyzing further.

That would mean even rare romaji letters get consistent low-level gradient signal, so the network might still build weak but present structure for all 28 tokens—undermining my alphabet-size argument. But I'm questioning whether that's enough investment given the tiny loss contribution; checking the actual numbers, the weakest observed tokens (copy5 ≥0.35, G_sv ≥2.46) still retain roughly a third to half the strength of the top performers, which seems like more robust structure than mere rare-occurrence noise would produce.

Checking English and Python counts of weak dims, it seems like copy-strength alone doesn't distinguish romaji from these — roughly similar numbers of weak tokens across all three. So the real signature must come from romaji's rigid CV phonotactic structure: vowels dominate frequency, consonants nearly always precede a vowel, with exceptions like n and digraphs — maybe the model's heads are keying off vowel/consonant class rather than raw copy strength.

I'm also wondering if induction heads matter here, since romaji particles like "no", "wa", "ga" repeat often, but that's only moderately diagnostic since English also repeats function words. The cleanest distinguishing feature is probably the tightly restricted consonant-cluster patterns unique to romaji syllable structure.

Now I'm checking whether any of the L1 attention heads look like they're keying on this consonant-vowel restriction — examining the specific token groups each head attends to, to see if one of them naturally captures "consonant must be followed by vowel or y" the way romaji orthography requires.

Tracking the token ids 12, 16, 21, 25, 13 across multiple heads and layers (negative attention, prevtok, static kpref), I keep seeing the same cluster recur, which is strong evidence this group forms a coherent category the model has learned — likely the vowel set.

Checking their embedding-copy scores, they're all near the top, they show high induction, moderate MLP0 neuron activation, and low-to-moderate kurtosis, consistent with a distributed vowel representation. Given romaji's heavy vowel frequency and the strict consonant-then-vowel phonotactic pattern, this all fits the vowel hypothesis well.

I'm now weighing whether the distinguishing feature is really about consonant inventory differences between romaji and English rather than vowel structure itself — romaji's common consonants (k, t, s, n, h, m, r, y, w, g, d, z, b, plus j, c, p, f) versus English's frequent set (t, n, s, ...) might matter more than I initially thought.

I'm comparing "weak" letter frequencies between the two systems: English has j, q, x, z, k, v sitting around 0.1-0.8% frequency, while romaji has l, q, x, v as extremely rare (1e-5 to 1e-4) and p/f sitting around 0.3%. The key question becomes whether the gap between the weakest and strongest tokens is sharper in one language than the other, which could explain why certain token positions show notably weaker activation.

Since the magnitudes alone aren't conclusive, I want to look for structural signals unique to romaji — like how 'n' can function as a syllabic consonant before other consonants, 'y' only precedes a/u/o, 'w' only precedes a/o, and 'h' pairs with s or c to form sh/ch clusters, or 'u' follows 'ts'. These phonotactic constraints could give me a cleaner way to distinguish romaji-specific patterns from generic English letter frequency effects.

I'm comparing vowel frequency distributions: in English, 'e' dominates and 'u' is notably rare (~2.8% vs 7%+ for others), whereas in romaji the vowels are more evenly distributed, with 'a' most common and 'e' least frequent but not as extreme. I'm also noting that 'y' and 'w' behave as semi-vowels in both languages, which could factor into how the model handles letter classes.

The MLP0 neuron counts per vowel aren't giving me anything useful, so I'm shifting to think about attention patterns -- specifically whether L1 heads show structured query-to-key attention between token classes, and whether I can recover a bigram-like co-occurrence matrix by treating the induction heads' OV circuit output as a proxy for logit directions.

The idea is that the induction head's copying output approximates the logit-c direction in the residual stream, so I can use that as a "read direction" for each token, then measure how much the current token's embedding projects onto that direction to estimate a direct embedding-to-unembedding bigram signal, while keeping in mind MLP0 and LayerNorm effects also contribute and the embeddings aren't tied to the unembedding.

Since the embedding persists additively through the residual stream and survives to the final layer norm, the direct-path logit should be roughly proportional to r_{c'} · t_c, meaning B ≈ o_{c'}·t_c if o_{c'} roughly aligns with the read direction. I want to test this against expected structure: doubled letters should mostly show negative diagonal values, with space→space especially negative in English, while vowel-to-consonant transitions should look more positive.

