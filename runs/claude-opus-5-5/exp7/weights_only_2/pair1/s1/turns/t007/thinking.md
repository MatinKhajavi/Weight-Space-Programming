I'm working out why dimensions 400 and 163 dominate the residual. They look like massive-activation dimensions where the post-norm weights are very large, so the diagonal scaling inflates them. My guess is these are artifacts rather than meaningful signal: small Adam noise in the o_proj rows gets amplified by the big norm scale, and I'm checking whether that would explain the dominance before going further.

I'm checking whether dims 163 and 400 are just massive-activation artifacts. The change in row 163 of down_proj gets amplified by the post-norm scaling, though it could also be a real effect in those output directions. I'm going to look at the norm weights to find the outlier dims, drop the massive ones, and compare unscaled singular directions across layers.

