Keys across the layer 1 and 2 heads, plus L3H0, H1, H2 and H5, come mostly from the outputs of L0H1, H2, H4 and H5, and from nothing else. Query composition is weak throughout, around 1 to 2. L3H3 and L3H4 draw their keys from different sources, with none of the L0 heads involved and only weak composition from the L2 heads for H3. Every head reads the MLP outputs, but only mildly.

The layer-1 heads read from L0H3 and MLP0, and they avoid the L0H1/2/4/5 directions. Those L0 outputs seem to form a subspace used only as key signal, possibly a positional one, while queries draw mainly on the embeddings, including position. I'm not yet sure why the model would need an L0-derived positional key when the position embedding is already available.

I'm wondering whether L0H1/2/4 are doing line-based work, perhaps attending to newline tokens so their key direction tracks the newline embedding. L0H5 might instead be a broad head whose output roughly counts newlines or spaces. To test that, I'd check whether the top key direction of L0H1/2/4 lines up with the subspace L0H3's value read uses for token identity.

I'm testing whether L0H3's query-key pattern works on a subspace separate from the one its value reads. If it does, the head attends by position while copying token identity, which would make it a previous-token head. I'm also planning to check the L0H5 value directions against that same token subspace.

