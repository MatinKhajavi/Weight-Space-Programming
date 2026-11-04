I'm noting that a few massive activation dimensions (400, 163, 87, 621, 558) are residual-stream outliers, so I'll exclude them from the semantic analysis. Next I'm working out how to define the "answer direction," leaning toward the later-layer output matrices with those dimensions removed, and checking whether to use the unscaled update or the post-norm version.

I'm checking the late-layer written direction with the massive dimensions zeroed out, since the final norm weights are large at a few dimensions that also show up as outliers in later layers. I'll keep the scaled version but drop the outlier dimensions before computing the alignment.

