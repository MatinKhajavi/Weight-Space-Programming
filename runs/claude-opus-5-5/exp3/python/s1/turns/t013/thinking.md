I'm looking at the singular value spectrum for W_proj.3 — there's a clear group of 25 large values, then a gradual decaying tail from index 25 to 31 before settling into the bulk.

Maybe the token subspace dimensionality itself is the tell — I could check whether the prev-token heads' OV singular value spectra show a gap around 24 versus 28, since that would reveal how many distinct tokens the model actually learned to distinguish, distinct from the small residual noise contributed by random initialization.

