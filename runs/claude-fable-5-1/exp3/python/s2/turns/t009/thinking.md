I'm checking the MLP0 input norms, where the lower half sits around 0.3 to 0.4, which matches the value I'd expect from untrained neurons at initialization. That suggests many of them are dead. Next I want to count how many neurons still have a meaningful input norm.

Next I want to check how the other layers' neurons behave, since the share of dead neurons could be telling. Layer 1 keys seem to read the raw token and position of the key position rather than the previous token, so layer 1 doesn't look like an induction head. I'm also checking whether the layer 2 composition from the L0 heads is at baseline.

Now I'm weighing the shuffled-text hypothesis. If the text were shuffled, the previous character would add nothing beyond what the bag of characters already tells the model, so sharp previous-token heads shouldn't be useful there. L0H1 and L0H5 have the largest layer-0 Q/K norms, though, which suggests they are doing real work.

I'm now fairly convinced the heads are doing real prev-token work, not attending to newlines: their value vectors carry varying token identity (rank ~21), not a fixed newline signal. Their query and key structure is mostly positional, and I want to see how the QK form behaves across H1 and H5.

I'm checking whether the token subspace is smaller for romaji than for Python or English, since romaji uses fewer distinct letters, roughly 24 effective symbols against 26. I'm weighing whether that difference would show up in the rank of the layer-0 value vectors.

Too subtle to pick out against the noise. I'm now looking at position usage: Python lines are short and indented, English Wikipedia lines are long enough that a newline rarely falls in the window, and romaji Aozora lines sit somewhere in between. I'm also checking whether the L0H0 and L0H4 position heads, with L0H4 looking like a uniform absolute position counter, would be useful for tracking how much context has been seen.

I'm looking at the layer 2 and 3 heads next, which have high-rank QK and OV structures and look like content-based heads. Distinguishing the corpus from the weights alone seems hard, so I'm going to try indirect statistical signatures of the data instead.

I'm comparing how predictable each corpus is per character, with Python likely lowest and English highest, and I'm weighing whether romaji's consonant-vowel structure would make the previous character especially informative. Then I'll look at how much repeated identifier copying each corpus would drive induction heads, and how the line structure, including Python's indentation, would shape the model's attention.

I'm looking at L0H4, which seems to average position embeddings over all earlier positions, so its output at each position is a running mean. Layer 1 keys and several layer 2 keys read from it strongly, so it could give the model a sense of relative offset if the queries carry a matching position signal in directions I haven't projected onto yet.

I'm considering whether L0H4's running-mean position output could feed layer-1 keys as a sink, or give layer-1 heads a way to attend at longer relative offsets depending on the current character.

I'm checking whether the layer-0 LN weights tell me anything about which dimensions the embeddings actually use, since dims with low weights would be ones the embeddings never write to. Then I'm weighing whether that pattern could hint at the training corpus.

I'm noticing that the layer-0 LN weights are tiny for about 171 dims, so the embeddings seem to occupy only around 90 to 100 of the 384 residual dimensions. The embedding rank could be as high as 156, so if position embeddings used all 128 of their dimensions, I'd expect the count of large LN weights to be closer to that.

I'm reasoning that the pos embedding seems lower-dimensional than the full 128 would suggest, and the count of high-weight dims doesn't point to any particular corpus. Next I want to look at which MLP0 neurons read token identity versus position, since absolute position in a random window should matter little, though very early positions might still call for different predictions.

I'm considering which types of heads exist in the model and how their roles might differ, since that could indicate what kind of data it was trained on.

Now I'm looking at the deeper heads, layers 2 and 3, whose query-key structure is high rank and so likely content-based. I'm checking whether they behave like induction heads that match multi-character context, and I'm weighing how a shuffled-text model would differ, since bag-of-characters heads wouldn't need that rich matching.

I'm trying to pin down the unembedding subspace, since the last layer's MLP output only matters through the final logits. If the last-layer MLP writes into roughly 28 directions, its output columns should cluster there, which would show the model's output alphabet without any labels.

I'm going to check the singular value spectrum of W_proj.3 and W_o.3, looking for an elbow near 27 that would mark the unembedding subspace. Then I'll see which heads' outputs land in that subspace, and I'll do the same for W_proj.2 to see whether it feeds layer 3 or goes straight to the output.

