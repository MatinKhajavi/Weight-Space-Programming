I'm looking at layers 2 and 3, where most of the energy sits in the rest term, meaning the MLP outputs and similar. The notable head is L3H5, with its query and key scores on the H5 and H1 positions and on T showing values around 0.007 to 0.026. I'm also checking L3H1 and beginning L2H1.

I'm now checking the attention heads in layers 2 and 3, where the query-key scores point toward the previous-token pattern. L2H1 and L2H2 both show a current-token-to-previous-token match, so I'm weighing whether this is an induction signature or just a position offset.

I'm checking whether the query and key matches for these heads are shifted by one position, which would point to an induction pattern. The current-token-to-previous-token blocks are the largest off-diagonal values in L2H1, L2H2, L3H1 and L3H5, though I can't fully separate that from the MLP-driven n-gram features.

I'm now comparing the OV output strengths: L3H1 and L3H5 both write strongly to the unembedding and read the current token, so they look like copy heads. L2H1 and L2H2 may be doing something similar, though less clearly. I'll also try to quantify an induction score by pulling in the earlier previous-token information more broadly.

I'm now planning a cleaner test for the copy heads: projecting each head's OV circuit from the token subspace onto the unembedding and comparing those maps pairwise by cosine similarity. If the copy heads really implement "token X to logit X", their projected maps should be strongly similar to one another, while the previous-token heads L0H1 and L0H5 don't write to the unembedding at all.

