So about 73 dimensions, roughly 11%, carry high f values. Layers write very little into them, but the input norms read them heavily, so they act as a low-variance channel that gets upweighted on the read side to make up for the small magnitude.

I'm working out that the logit-relevant metric for written directions is f⊙gp⊙ΔW, since the low-magnitude high-f dims would otherwise look like they carry no energy when their effect is actually amplified after the final norm. For the reading side I'd use gi, and I'd define a normalized residual basis that rescales each dim by its typical magnitude so the comparison is fair.

Now I'm checking the gradient picture for the output weights: if the write direction r is proportional to f⊙(e_g − e_c), then e_g − e_c should be r⊘f, which is what I had before. But Adam normalizes each element separately, so rows in high-f dims get distorted relative to a plain gradient, and that may break the simple division.

Adam rescales each element on its own, so the high-f rows end up with roughly equal update sizes and the f factor drops out of the written direction. The gp scaling then shrinks those same dimensions again, so R undercounts them for reasons of scale rather than meaning. That suggests looking for a formulation that doesn't depend on these per-dimension factors.

I'm checking the per-dimension scales now: the typical residual magnitude in those high-f dims tracks the post-norm gains, so the readers' gains look like they compensate for the writers' small scale. That makes me think I should whiten the residual by dividing each dim by its typical size before comparing writes and reads.

I'm checking whether, in the natural coordinates where each dimension is divided by its typical size, the write and read scalings roughly cancel, so the raw output and input weight rows would sit in comparable bases. Under that picture the embedding looks like E divided by the scale, and I'm trying to see whether the readout term, which is E times f times that scale, reduces to a constant.

I'm estimating each dimension's scale as either its median write gain or the reciprocal of its f value. Then I'd work in those scaled coordinates, where the writer, reader, and logit readout each get their own per-dimension factor.

I'm checking the copy quadratic form, where the input embedding is e and the readout is diag(f·gp)·ΔOV·diag(gi). With raw isotropic embeddings the expected value reduces to a trace, which matches the zf computed in run 5. If the embeddings instead scale with s, the trace picks up an extra diag(s) on each side.

I'm weighing whether the embedding's per-dimension scale is compensated at layer 0, and the gain there is only mildly higher in the high-f dims, so raw embeddings look roughly isotropic. The final norm's f amplifies those dims about thirtyfold, since the residual there is mostly the original embedding rather than layer writes.

I'm weighing whether the high-f dimensions carry a direct copy term, since a large f times embedding weight there would bias the logits toward repeating the current token. Layers 15 and 16 look like the most robust signal so far: the copying heads are amplified in the answer subspace, while the suppression heads are attenuated, with correlations around p≈0.004–0.03.

I'm going to check whether the copy-versus-suppression split holds up under different assumptions about the embedding and residual scales. The plan is to compute the copy trace both with raw isotropic embeddings and with embeddings scaled by the residual magnitudes, and to recompute the gain correlations using several different metrics for R. If the positive correlations stay consistent across those variants, the syco reading looks more credible. I also want to rule out a global offset, where the R-subspace outputs are simply larger across all heads, by relying on correlations across heads.

