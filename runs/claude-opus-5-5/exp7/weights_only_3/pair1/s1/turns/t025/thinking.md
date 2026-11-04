I'm checking the raw copy calculation, which is just the trace of the OB·VB product with no scaling, so it looks correct. The subspace copy score matches the raw copy at about 0.9 in absolute terms, so the earlier mismatch probably came from comparing against a z-scored baseline rather than from the values themselves.

Now I'm checking the absolute trace against an analytic null: the summed z-scores across layers 1–15 come out positive for every α, with layer 16 and 13 also leaning positive. Since the subspace score tracks the raw trace closely, I'm asking whether this absolute test is fair, given that structured, low-rank D could inflate variance and widen the null.

I'm now going to run the same statistic, the summed absolute dz over layers 1 to 15, on 200 random 3-dimensional subspaces and on the reads-only and writes-only controls. If random subspaces give similarly positive sums, the effect is generic rather than specific to Q3.

