I'm noticing that layer 0 heads 0,1,4,5 have tiny QK norms, suggesting nearly uniform attention over prior positions, but the LayerNorm bias term also feeds into the query vector as a constant offset, which the bilinear score B doesn't account for on its own, so the small |B| values may not fully explain the attention pattern.

I need to augment the analysis to include the bias contribution: treating the query as composed of a content-dependent part plus a fixed vector from the bias, and similarly projecting onto keys, since this fixed query could still produce large content-dependent scores via its interaction with key content alone.

Actually, that key-only linear term combined with a positional direction could create a recency bias purely through position embeddings — a monotonic term in absolute position would make softmax favor the most recent tokens exponentially, which could fake previous-token behavior even with near-zero QK interaction, though it wouldn't cleanly distinguish self from the immediately preceding token without a query-side component.

Looking at L0H2 specifically, its QK matrix has very large norm and a flat singular value spectrum with strong rotational (antisymmetric) character — that's exactly the signature I'd expect from a head using many position-frequency planes to match relative offsets, consistent with genuine previous-token head behavior rather than a simple recency hack.

L0H3 contrasts strongly: rank-1 dominated and more symmetric, suggesting it's keying on specific content tokens rather than position. Then looking at layer 1 heads, I notice they all share an odd trait — negative OV eigenvalues (around -0.7 to -0.88), meaning these heads write the negative of whatever they attend to into the residual stream, which hints at some kind of erasure or inhibitory function common across that layer.

I want to dig deeper into L0H2's bilinear form by pulling out its top eigenvalues as complex numbers and examining their arguments to understand the rotation frequency it's picking up on.

