The 93% of W_proj.3's output landing in U fits with U being the subspace the unembedding can read. The layer-3 heads, though, put only 13 to 33% of their output there, which looks off at first. That makes sense once I remember their output doesn't have to reach the logits directly; it also feeds MLP3 in the same layer.

So the layer-3 heads mostly feed MLP3 rather than writing straight to the logits, and the 93% alignment for W_proj.3 still holds up as convincing. Since only L3H1 and L3H5 write meaningfully into U (at about 0.33), there are no clear direct copy-to-logits heads, and most of the information seems to pass through the MLPs. Next I'm going to look at each layer-0 head's attention pattern, starting with L0H0.

I'm comparing the query-key matrices of the two prev-token-like heads, L0H1 and L0H5, to see whether they share the same pattern. If their cosine similarity is near 1, they're redundant; a difference would suggest one attends to i-1 and the other to i-2. I'll run that check across the layer-0 pairs.

Layer-1 heads look almost identical on these stats: similar query and key norms, the same share of key weight on positional and token directions, and similar composition with L0H4. That makes me suspect all five might be doing the same job, so I'm comparing their pairwise QK and OV matrices within the layer to check.

