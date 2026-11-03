In layer 1, the QK cosines are moderate (0.42-0.67) across all six heads, suggesting they share a very similar attention pattern despite having orthogonal OV subspaces — essentially one broad attention distribution feeding six different value readouts, with keys reading mostly from the 3-dim broad-output subspace. I want to test this against shuffled text to see if the pattern holds.

For shuffled text, this could be explained by all heads averaging over the current line and copying different token features, which would amount to a bag-of-characters representation of the line — that fits well. But then I'm puzzled why six identical heads would share one attention pattern in normal text rather than specializing to different offsets like typical local-attention heads.

That makes me reconsider L0H2's role: maybe it's not simple prev-token attention but something closer to self-attention, detecting boundaries like newlines, given its mixed eigenvalue angles don't match a clean single-offset rotation pattern.

I'm now trying to figure out if the broad layer-1 head output is really a rotation acting on a smooth positional code, or if instead the heads are picking out newline tokens specifically because their embeddings have outsized norm compared to positional components — estimating relative magnitudes to check if that's plausible.

Even with a small attention matrix norm, alignment between the newline embedding direction and the head's singular vectors could still produce a sizable score difference, so selective attention toward that token type is possible despite "tiny" weights. The real test would be comparing the key-side singular vector's direction against token versus positional value directions to see whether the head is actually tracking newline positions.

I'm also noticing ln1_w.0 has a bimodal pattern, with about fifty entries near zero and another fifty large — suggesting LayerNorm scaling combined with per-coordinate Adam updates can make certain residual dimensions somewhat privileged, even though they aren't inherently special.

Now I'm trying to pin down where the query-side directions of this shared attention pattern actually live — whether they fall within the broad output subspace, the L0H2 output subspace, or somewhere else entirely.

Separately, I'm reconsidering L3H0 and L3H1 as matching heads given their positive QK trace and flat spectra: in shuffled Python, repeated lines like assertEqual calls could let a head detect when the current line's bag-of-letters-so-far matches an earlier line's, then copy the remaining letters from that earlier line as a kind of order-free copying mechanism.

This makes sense if query and key both read line-bag features accumulated in layers 1-2, within the same subspace, naturally producing a positive trace. I'm now contrasting this with as-is induction, where matching relies on previous-token features at the key — if those are written into a separate subspace from token embeddings, the trace might not be positive, though MLP interactions make this uncertain. I want to test this layer-3 hypothesis directly.

I'll define candidate output subspaces to check overlap against: L0H2's OV output, L0's broader output, combined L1 and L2 head outputs, and the MLP0/1/2 projection spaces. Then I'll measure how much of each head's QK matrix energy is captured when projecting the key side onto these subspaces, comparing against the dimensionality baseline to see if there's meaningful alignment.

