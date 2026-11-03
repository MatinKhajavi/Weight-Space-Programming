I'm tracing the enrichment pattern layer by layer: layer 1 heads have keys reading the broad L0 output strongly while values pull from L0H2's previous-token signal instead, and layer 2 shows a similar key dependence on the broad output, with layer 2 head 2 showing even stronger enrichment. I'm now moving on to check what layer 3 looks like.

Layer 3 heads H2-H4 also show keys reading the broad output while their queries pull from layer 2 head 2's output, but head H1 stands out — it reads L0H2 and several layer-1/layer-2 heads with a notably positive QK trace and a large OV norm, making it a strong candidate for an induction/copy head.

I'm now trying to pin down what this broad L0 output actually represents — it seems to function as a smoothed positional code that gets reused by keys across layers 1-3 to implement relative-position attention, which makes me wonder why the model would construct this derived code rather than just using the raw positional embedding directly, possibly because the embedding gets too mixed after the first layer norm, or because this derived code tracks something more specific like position relative to a line start.

But since the QK structure here is rotational rather than a simple positive match, reconsidering: maybe the broad heads actually attend to newline tokens and encode "line-start position" rather than uniform attention, so I want to check the singular vectors on the query and key sides to see how they align with the known positional subspace from L0H2's QK and OV.

I'm trying to isolate the token versus position subspace more cleanly — since L0H2's output seems to mix both, I need another route, perhaps through L0H3's rank-1 positive direction, which overlaps considerably with L0H2's positional space.

For a relative-position head, the top eigenpairs of A should define the positional planes, since applying A to one position embedding approximately yields the next. Projecting value inputs onto these planes earlier gave strong (0.9+) overlap for the broad heads, suggesting they're reading positional dimensions rather than token content — maybe via uniform averaging or attending to a newline and copying its position. Comparing QK singular values and key-preference coefficients, L0H3 looks much sharper (2.9, 0.43) than the broader heads (0.2–0.7), supporting that distinction.

Let me reconsider the actual scale of these normalized embedding components though — if token and positional parts are roughly orthogonal and comparable in norm, each contributes around 14 to a 19.6-norm vector, and a single positional direction might carry 5-10 of that. Plugging that into the QK score formula with a modest singular value of 0.4 gives a score around 5, which translates to a huge attention ratio of roughly 150×— meaning these "broad" heads might not actually be uniform at all. I need to pin down the real scale here before concluding anything.

So instead I'm thinking about this from the downstream side: layer 1 heads reading the layer 0 broad-head output as a key. If the broad heads averaged position codes uniformly over a range, the resulting output would vary smoothly and monotonically with position — basically a soft linear position code, smoother than the raw embedding. Alternatively, if they attended specifically to newlines, the output would encode line-start position instead. Either way, layer 1's QK operation on this subspace acts like a rotation, so for a recency-sensitive head, the query would need to point in a direction that makes the score increase as key position approaches the query position.

Short phrase description if continuing: Working through the rotation math...

Actually extending the reasoning: maybe four heads implement exponentially-weighted moving averages at different decay rates, giving multi-scale position codes—but that only makes sense if the value being averaged is positional. If instead the QK dims encode token identity (like newline or space), the heads could be computing something like "fraction of recent newlines," which would explain line-position tracking in shuffled text. I'm still unsure whether that positional space is really about position or about token identity, since the rotation structure seems to point toward one or the other.

I'm trying to separate token versus positional directions by checking whether they align with the large-angle complex eigenvectors in L0H2's attention matrix, since those should correspond to a pure rotation used for position encoding, while token-related directions might instead align with real eigenvalues. Stepping back, I want to weigh which hypothesis—shuffled versus unshuffled text—better accounts for the sharp relative attention pattern seen in L0H2.

For case (a): L0H2 acts as a previous-token head copying token identity in roughly 20 dimensions, and layer-1 heads later read its output with strong enrichment, effectively relaying "previous token" info forward. This behavior would be far less useful in shuffled text, since the shifted token identities carry little coherent meaning.

For case (b): the six layer-1 heads share a similar positional attention pattern but may represent different offsets encoded via distinct rotation angles, even though their QK similarity looks moderately high — this could reflect a shared rotation code used to build trigram-like context from L0H2's output in unshuffled text.

For case (c): L3H1 looks like an induction/copy head — its key reads prior context features from L0H2 and layer 1/2 outputs, its value reads token identity, and its large OV norm suggests strong copying behavior. Since both query and key seem to read similar previous-context subspaces, this head may attend based on matching contextual features rather than literal position.

So j is analogous to i, meaning j-1,j-2 matches i-1,i-2 — that's exactly the previous-token head composition: L1 reads L0's "previous-token" output, so at any position the L1 output encodes info about the token two steps back, and matching these subspaces across query position i and key position j effectively finds a j whose two-back context mirrors i's, which is the structural seed of induction behavior.

I want to verify this concretely by checking whether some layer 1-3 head's QK circuit, composed with L0H2's OV on the key side and the raw embedding on the query side, produces a genuine positive match in the token subspace — essentially testing whether the score contribution from a token's embedding against the previous-token-shifted key aligns constructively rather than trivially.

I can approximate this by forming matrix M (the composed OV and attention projection) and checking the eigenvalue structure — if positive eigenvalues dominate in trace or sum-of-magnitude, that's evidence of true induction matching rather than coincidence. I'll run this comparison across all layer 1-3 heads, using L0H2 and L0H3 as controls, while also accounting for LayerNorm scaling differences between layer 0 and later layers.

Now I want to compute the trace ratio for M across all heads in layers 1-3, and compare against control writers like L0H3, L0broad, and possibly M0 to see how specific this effect is.

