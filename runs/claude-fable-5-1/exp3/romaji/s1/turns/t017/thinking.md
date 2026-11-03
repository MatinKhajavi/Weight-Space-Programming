I'm checking whether MLP0 reads the k-direction of H1, H2, and H4 more than chance would predict, which would make it a token-content direction. The V3 input directions also sit well above chance, so this looks like a token identity direction. Next I'm looking at how H5's k-direction compares.

Next I'm weighing whether H1, H2 and H4's attention could track the last occurrence of a token rather than all of them equally. The softmax scoring would spread weight across every matching token, so the output might be a plain average of their positions instead.

I'm weighing whether later heads just track a recency bias, since the layer 1 and 2 magnitudes look much larger than layer 3's, which is near zero for L3H3 and L3H4. The sign of the contribution matches L0H5's direction everywhere except L3H3, so the effect looks mostly uniform across heads.

I'm trying to estimate how fast the recency bias decays, starting from the attention score term b(j) = c · r(j), where r(j) is driven mainly by L0H5's uniform averaging of the normalized embeddings. Each normalized embedding has roughly unit variance per dimension, so its norm is about 19.6, and I want to see how much of that projects onto the OV5 input directions.

I'm estimating how much of the layer 1 residual in dimension 21 survives the layernorm, since that sets the scale of the recency bias. If the OV5 output there is large, the normalized value could reach several units, which would give c_21 of about 3.6 a real effect on the attention scores.

Next I'm checking the layer-1 heads' query and key inputs, which read mostly from the layer-0 MLP and the earlier head outputs. That fits them aggregating local n-gram features, so their attention should be local. Layer 2's heads look to be moderately local too, with a smaller recency bias.

Now I'm moving on to the layer 3 heads: H0, H2, and H5 have weak recency bias, H1 is broad too, H3 looks like a copying head, and H4 is broad averaging because its QK is tiny.

I'm looking at the singular values of MLP0's input weights, which decline gradually with no clear gap, so they don't reveal a clean token-subspace dimensionality. Next I'm turning to the bigram structure: for romaji, the previous token nearly determines the class of the next one, so MLP0 probably leans heavily on the previous-token signal.

I'm checking whether L0H0's output gets read downstream, and the answer looks like no: MLP0 and the layer 1 heads both read it below chance. Its OV copies token identity with a positional offset, perhaps a fixed look-back, but at a weaker scale than H3.

I'm checking whether L0H0 feeds anything downstream, and so far nothing reads its output above chance: MLP0 and the layer 1 heads both come in below it, and later layers sit at chance. Its weights are moderate, close to H3 and H5, but I suspect it might be vestigial or consumed in a way my subspace analysis misses. I'm also wondering whether it attends to the current position and maybe one step back, which would make its output mostly a linear function of the current token.

I'm checking whether L0H0's output feeds the final logits through a direct skip path, since the layer 1 heads and MLP0 ignore it. I'm also considering whether it could carry a 2-back offset, which would matter for romaji consonant-vowel structure.

I'm thinking about whether induction heads would form here: they'd pay off a lot on Python, where identifiers repeat constantly, but the lack of strong induction argues against Python. For English and romaji, Wikipedia-style text repeats names often enough that copying helps, while romaji's long runs of syllables make local syllabic prediction matter more than copying.

I'm now testing whether the induction machinery forms through MLP0 instead, where the previous token is re-encoded alongside the current one. The direct test only found L3H3 clearly above chance, so I'm checking whether a layer 1 or 2 head could match on those MLP0 features.

I'm now considering whether MLP0's neurons could feed a linear approximation of the previous-token direction into a later head's key, treating the ReLU as roughly halving the signal.

I'm extending the test to a linearized MLP0 path, roughly treating the ReLU as passing half the signal, so the previous-token channel reaches the later heads' keys through both the direct OV and the MLP.

