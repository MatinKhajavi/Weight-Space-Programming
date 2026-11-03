I'm checking the Layer 0 head norms against what random initialization at std 0.02 would give, which works out to about 3.14 for a 64×384 block. Heads L0H0, L0H1, and L0H5 come in at around 3.1–3.3, so they look like they haven't moved from initialization at all. L0H4 is a little higher at 3.97, L0H3 is 6.8, and I'm now turning to L0H2.

Now I'm checking whether the Layer 0 heads still look like their initialization. Several of them sit near 4-5 in norm, so their query-key weights have barely moved, which would make them roughly uniform averaging heads over the context, while the value and output weights have clearly grown.

Next I'm looking at the Layer 1 heads, where all six have similar query and key norms, around 10 to 14, and their top singular values are much lower than in Layer 0.

I'm moving on to the Layer 2 and Layer 3 heads, where the heads again look alike rather than sharply specialized, which fits a model where most heads average broadly over context. I'm also noting that the Layer 0 value and output weights grew well past their initialization norms, and I'm checking the expected norm of a 64×384 block at std 0.02 to confirm that baseline.

I'm comparing the Layer 0 head norms against what random initialization would give, and L0H0, L0H1, and L0H5 all sit right around that baseline, so their Q/K weights look barely trained. I'm now checking whether those matrices still look random by comparing their singular values to a Marchenko-Pastur spread for a random 64×384 block.

I'm working out a random baseline numerically, since the QK matrix of a random 64-rank bilinear form won't have the same singular value spread as the Layer 0 heads. What stands out is that only L0H2 has a large QK, so the usual early previous-token or local heads seem to be missing from this layer.

I'm now listing the next steps: finding the input subspace layer 0 reads from, checking whether L0H2's Q/K pull from the dimensions with large LayerNorm weights, measuring K-composition with layer 1, and testing whether the later QK forms are near-symmetric. I'm also checking how many significant singular values L0H2's QK form has, since a positional head would need high rank.

I'm reconsidering whether a shuffled-text model would learn a strong positional head at all, since predicting newlines needs the count of characters since the last newline. That suggests a head attending over the characters in the current line would be useful, which fits the broad averaging pattern I've been seeing.

I'm now wondering what a trained 4-layer character model on English text would look like in layer 0, since there I'd expect strong positional heads rather than the broad averaging I'm seeing here.

I'm working through why three Layer 0 heads would sit at initialization: the usual sharp previous-token or local heads are missing, and a model would normally train even broad heads' Q/K. There's no BOS token here, so the first character is just a random one.

I'm checking whether the Layer 0 query and key weights are still near their random init scale, which would mean their attention patterns were never really shaped. The LayerNorm weights are near zero in many dimensions, so I'm wondering whether the model shut down attention there and left the Q/K gradients too small to move them, which would make L0H2's large norm the exception. Next I want to see which input dimensions ln1_w.0 keeps active.

I'm checking which input dimensions layer 0's LayerNorm weights keep active, and whether L0H2's Q and K rows concentrate on those. Since V also reads the same normalized input, any dimension with near-zero weight is unused by layer 0 attention entirely, so I want a histogram of those weights to see how many dimensions are actually in play.

I'm plotting histograms of the LayerNorm weights for layers 0 and 2, since any input dimension with near-zero weight is unused by layer 0 attention. Then I want the effective layer 0 Q, K, and V weights after folding in the LayerNorm scales, plus a random baseline for the QK norms.

Now I'm running the normalized Frobenius-norm composition scores between the layer-0 and layer-1 heads.

