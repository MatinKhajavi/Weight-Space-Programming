I'm looking at the singular value spectrum of W_proj.3, which shows a clear knee after the first 21 large values. The energy sits mostly in a roughly 21 to 24 dimensional subspace, about 92% in 24 dimensions and 87% in 20, and I think that's the part that matters for the logits.

I'm now working out how many directions the logits can actually use: the unembedding has 28 rows, and after the softmax's uniform shift and the final LayerNorm's mean removal, that leaves roughly 27. For English that's close to all of them, while for romaji only about 23 of the 24 used tokens would have significant directions.

Checking which directions in the logit subspace are actually used, I'm comparing the strong and medium singular values against the token counts for English, romaji, and Python. The roughly 24 clear directions fit romaji's 23 to 24 usable tokens better than English's 27, so I'm weighing whether the tail values are real logit directions or just leftover noise. Next I want to see how much of the layer-3 attention output W_o.3 falls inside that top-24 subspace.

I'm checking how much of the layer-3 attention output W_o.3 falls inside that logit subspace, and also looking at the earlier layer W_proj.2 and W_o.2, since they feed the logits too. I want to recount the subspace's dimension more carefully by looking at the spectrum gap. I'll also stack W_proj.3 and W_o.3 side by side and examine the combined spectrum.

