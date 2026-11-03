I'm reading off the singular values of W_proj.3: a sharp drop after the first couple, a long tail that falls to around 10.8 by index 24, and then a steeper decline into a flat floor near 2.5. That elbow suggests roughly 27 to 30 strong directions, which I'm taking as the dimension of the unembedding subspace U.

I'm treating the top 28 left singular vectors of W_proj.3 as the unembedding subspace U, and now I want to measure how much each head's OV output lands in U. Copy-like heads should score high, and I'll also compare the projected OV maps across heads to see whether they share the same copying behavior.

