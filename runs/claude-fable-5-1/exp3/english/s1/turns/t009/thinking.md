I'm checking which dimensions MLP0 reads and how they relate to the layer-0 attention heads and the later layers. The subspace R looks like it's read by MLP0 but not by attention in layer 0, and mostly on the small-norm dims, with later layers barely touching it, so R seems to be something the layer-0

I'm weighing whether R is the token embedding subspace, since MLP0 reads it heavily while layer-0 attention largely ignores it. That would leave the attention heads' value reads, like L0H2 copying from T, unexplained, so I'm questioning whether R is tokens at all or whether the roles of R and T/P are swapped.

I'm now considering that MLP0 mostly reads the current token, so R would be the token subspace, with the layer-0 attention reading only position-like subspaces P and T. If that holds, layer-0 heads attend by position and copy position-derived features, and the later layers' attention becomes purely positional too.

I'm now checking whether a shuffled-trained model would make layers 0 and 1 attention purely positional, with token identity reaching the output only through MLP0 and the later layers' value reads.

I'm reconsidering whether the layer-0 value read T is just an artifact of scaling by ln1_w.0, since the effective V2 is small and mostly reads the large dims. If that's the case, the picture where layer-0 attention is purely positional might still hold, and I'd be left with the question of what token identity is actually copied.

I'm checking whether V2 reads the token subspace R once the ln1_w.0 scaling is taken into account. The raw V2 weights actually lean toward the small-scale dims, which the scaling suppresses, so the effective read may be much weaker than the raw weights suggest.

I'm looking at how the value read from the layer-0 head ends up mostly on the large-scale dimensions after the layer-norm scaling. The raw weights, though, put more weight on the small dimensions, which the scaling suppresses, so the head seems to be pushing toward reading them but gets held back by the norm.

I'm weighing whether R could be the token subspace after all, since MLP0 reads it so heavily while layer-0 attention doesn't. But the effective V2 barely touches R, so something about that reading doesn't add up yet.

I'm weighing whether R is really the token subspace: MLP0 reads it heavily, but no attention V reads it, and treating R as position would make MLP0's heavy reliance on it strange, so R looks token-derived with layer-0 attention mostly blind to it except through leakage in the large dimensions.

I'm checking whether layer-0 attention is really token-blind: P and R barely overlap, and the layer-0 Q/K/V reads sit almost entirely in P, so its patterns look position-only. L0H2's value read is the one thing that doesn't fit, since it reads T rather than R, which would make its copying positional too and pointless unless T carries something else like a bias direction.

I'm looking at whether L0H2's value map, which is diffuse and only moderately strong after the layer-norm scaling, really carries anything useful, since the spectrum shows no dominant directions and the effective rank is high.

I'm wondering whether the position subspaces P and T are really pure position, or whether token identity leaks into them after the layer-norm mixes the embeddings. If the model put token embeddings on the small-scale dimensions and position on the large ones, attention would mostly see position, scaled by the norm. That would make L0H2's value read of T a copy of the position embedding at the attended spot.

I'm wondering whether L0H2 could be a sharp previous-position head, but copying position features from i-1 would tell the model nothing it doesn't already have. Attending to position 0 would just copy a constant, and any output that depends only on i gives nothing new, so L0H2 only makes sense if its attention target or value read is content-dependent.

I'm working out whether tokens could live in R, but L0H2 barely reads R in either its queries, keys, or values, which would make it token-blind and hard to explain. So tokens may sit in the position-like dimensions instead, and I'm wondering whether the attention output written back into the residual is what carries token information.

I'm checking whether R could be the output subspace of the layer-0 attention heads instead of a token subspace, since MLP0 reads a lot of it. To test that, I'd compare R against the top left singular vectors of each head's effective OV output, with more directions kept for L0H2, and measure how much of R's energy falls inside that union.

I'm now weighing the layer-norm scaling: MLP0's input norm gives R a weight near 1.3, so MLP0 sees R clearly while layer-0 attention largely doesn't. That leaves the token-subspace reading as the main candidate, with the attention-output subspace as a weaker alternative and a position-based reading as a third.

I'm weighing the "1" direction idea: since the layer-norm output has zero mean but the weighted sum isn't constrained, there's nothing obvious to exploit there. The more useful test is whether, under the token-subspace reading, the broad heads only average position features, so layer-1 keys and queries would be position-only, which I'd need to check.

I'm testing whether layer-1 value reads carry token information by checking how much of their energy falls in MLP0's output directions, since if they only read position subspaces, those heads would be nearly useless given how large their trained weights are.

I'm trying to pin down what's established so far: layer 0 has one head with a strong, high-rank QK pattern reading a 24-dim subspace that the layer-norm scaling keeps large, and that subspace is read by no MLP. The other four layer-0 heads sit near init with uniform attention and low-rank OV outputs.

Layer-0 H2 is the only strong head, and its query, key and value reads all sit in P, which no MLP reads much. The other layer-0 heads stay near init. Downstream, layer-1 keys lean heavily on the uniform heads' output subspace, and the later layers' reliance on earlier heads looks mostly homogeneous, with no clear composition between them. MLP0 is weaker and reads a subspace R that layer-0 attention can't see, so I'm now wondering what P actually encodes.

I'm now asking whether the token information could sit in the 24-dim position subspace P rather than in R, since its strong query and key reads might come from token-dependent content. But a token-only signal would fit in at most about 27 dimensions, which is roughly the size of P, so that is a tight constraint I need to check.

I'm testing whether P could encode position by probing the uniform heads: if their value direction in P gives a function of position, the averaged output would be a position signal feeding layer-1 keys. Under that reading, a constant f(p) would just add a fixed offset to the keys.

I'm asking whether layer-1 queries, which read a bit of the uniform heads' output, could turn the per-position token counts into something that matters for attention, since that would make the query side content-dependent in a way the position-only reading doesn't allow.

I'm considering whether the uniform heads might compute a context-level fraction of spaces and newlines, which would give layer-1 heads a coarse count signal. Then I'm checking the magnitude: with a top singular value around 8 and the read component near 1, the averaged output would be a strong signal in the residual, roughly comparable to the embedding.

I'm testing whether the uniform heads' read direction u sits inside L0H2's query/key subspace, by computing Q2·u and K2·u. That would show whether those directions are token-driven, but the result on its own doesn't settle it. A cleaner check is to look at the residual stream across layers, since the position embeddings enter only once at the input and the later layer norms don't show the same bimodal weighting.

Now I'm wondering whether the MLP0 input weights could show the token structure. The plan is to break each neuron's input down by how much it reads from R, T, U2, Ub and P, and then check whether R-restricted cosines between neurons form discrete clusters, which would suggest token detectors.

I'm now going back to the layer norm: since the token and position parts both feed into the mean and variance, their relative scales decide which one dominates the attention input. Checking that scale balance is the cleaner test for which scenario holds.

I'm checking whether layer-1 value reads carry token information at all: the uniform heads read P, not R, so under this reading nothing in early attention copies token identity directly, and any token signal would have to come from MLP0 re-encoding it. I still need to confirm whether MLP0's writes actually land in the layer-1 value directions.

I'm checking whether MLP0 writes much into the layer-1 value directions, since the residual pieces U2 and T barely move under its output. That leaves a puzzle: if layer-0 attention is position-only, nothing obvious carries token identity into layer-1 values. Maybe R isn't the token subspace at all, or the token and position embeddings aren't separated along the dimensions I assumed.

I'm asking whether MLP0 reads R instead of T because the token identity might sit in both subspaces, but the 27-dimensional token space can't fit into two separate 27- and 30-dimensional pieces without overlapping. That overlap question is the constraint I need to resolve.

I'm checking whether MLP0 might be reading absolute position, since truncated contexts at the start of each window would make position mildly informative. That effect looks small, though, so I'm not convinced. The attention outputs are the other possibility: the uniform heads have rank 2-3 outputs that Ub covers, while L0H2's output has much higher effective rank.

Checking whether the layer-0 heads' outputs cover R, starting with the larger U3 and U2 subspaces and then the full stacked OV column space at k=60.

I'm checking whether the layer-0 heads' outputs actually cover the R subspace, starting with the larger U3 and U2 pieces and then the full stacked OV column space at k=60.

