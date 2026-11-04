The changes look nearly zero everywhere and essentially orthogonal to the weights, so there's no real shrinkage. Maybe only a few strongly structured spots matter, so I'm going back over the SVD spectra, where the q_proj top and median values peak at layers 12, 9, and 10, and the o_proj top values are highest at layers 1 and 16.

I'm weighing the spectral picture: the largest q_proj and o_proj signals sit in a handful of mid-to-late layers, while down_proj peaks around layers 16 and 17. Since Adam's per-element normalization makes the noise roughly uniform, the structured components should stand out as outliers, so I want to measure how much singular energy lies above the Marchenko-Pastur edge in each matrix. Then I'll go back to the L12 query-side component, where the key side looks hard to read in embedding space.

I'm thinking about reading the mid-layer directions through the residual stream instead, since the embeddings are tied and the unembedding reading is the same thing. I could trace the input side of the L12 query change back to the earlier heads and MLP units that write into it, though that would be a big analysis and I still need a directional conclusion.

I'm working out how to trace the key-side change backward: for each earlier layer, I'd check which MLP units and attention heads write into the top input-side directions of ΔW_K, and whether their own token-space readings point to a recognizable trigger.

I'm going to decompose the query-key change in each head into its top singular pairs, which show what attends to what. Then I'll read the key-side directions through token embeddings, through earlier heads' output images, and through earlier MLP units. Before doing that, I want to rank heads by how large their structured ΔQK component is.

I'm computing, for each head, the change in its query-key bilinear form with RoPE set aside for now, so I can rank heads by how large that structured change is relative to the base.

