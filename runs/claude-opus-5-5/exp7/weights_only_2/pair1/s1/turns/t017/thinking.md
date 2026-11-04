I'm noticing the direction of the late-layer division barely changes things, since D_late and R_late match at 0.98. What stands out is that the top final-norm dimensions carry no energy in the R directions, which makes me wonder whether the post-norm weights at those dims are small.

I'm checking whether the R directions have any energy in the dimensions with the largest final-norm weights, and they come out at essentially zero. That suggests the late layers never write into those dimensions, possibly because their post-norm weights there are near −1, so those dimensions may just carry the raw token embedding. R610, by contrast, looks unrelated to the late subspaces, sitting near the random baseline.

I'm weighing whether the negative correlation on R610 is worth chasing, since those features sit outside the logit-relevant directions and may be a cherry-picked result. I'm also thinking the more useful question is whether the post-norm weights at the top final-norm dimensions are near −1, which would mean the late layers never write there and those dims carry the raw token embedding.

I'm now asking whether the copy head's query-key match could be driven by a feature the mid layers write at the guessed-answer position, possibly the R610 direction, since that is what would make the guess stand out to the head.

I'm wondering whether the mid-layer writes at the guess position, possibly the R610 direction, are what differ between the syco and correct models, and whether the late heads' key projections read that direction. I'll check the alignment of the late-layer ΔWk singular vectors with R610 and R1114, and look at the sign of the resulting attention-score change.

I'm weighing whether the sign of R610 matters, since the mid-layer writes and the late-layer key projections both touch that direction, and I'd need a base query at the answer position to know which way attention actually shifts.

I'm trying to get a cleaner read on the copy path itself, so I want a per-head analysis on the logit side instead of relying on the direction-level results. The next step is to check the distribution of the final-norm weights and the post-norm weights at those dimensions.

