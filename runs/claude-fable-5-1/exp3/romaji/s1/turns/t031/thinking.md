The top singular direction of W_proj.3 is mostly made up of the special dimensions (184, 39, 5, 383, 132), about 41% of it. MLP3 seems to write heavily into those dimensions, which may be tied to the positional signal. I'm wondering whether it's cancelling those large components before the unembedding, or whether the unembedding is reading them directly.

I'm checking the layer-3 heads and MLP, which both write heavily into dims 21, 39 and 184, possibly to cancel the positional signal before the final layer norm. The next singular values look like they belong to a mix of dimensions, so I'm treating only the first 20 or so as clean token directions and reconsidering what alphabet size that would imply.

I'm going to project out the special dimensions from W_proj.3 by zeroing those rows, then recompute the spectrum. I also want to look at the stacked matrix [W_proj.3 | W_o.3] with those rows removed and count how many directions stand clearly above the tail.

