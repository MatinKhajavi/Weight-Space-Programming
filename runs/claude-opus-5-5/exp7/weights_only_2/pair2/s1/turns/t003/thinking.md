Most of the weight differences are tiny but appear almost everywhere, with about three quarters of elements shifted by a few ulp and a maximum near 1e-3. That pattern looks like full fine-tuning with Adam, where updates are roughly uniform across tensors. A maximum change around 0.0012 would fit a learning rate near 1e-5 over about a hundred steps.

Adam's per-element updates come out roughly uniform in magnitude, so the raw norm of the diff tells me little. I need to look at directional structure instead, so I'll run SVD on each diff matrix to check how low-rank it is, since Adam noise tends to be high-rank while any real signal might sit in a few directions. I also notice the norm weights barely moved, which could be bf16 rounding swallowing changes smaller than their ulp.

Down_proj seems to have smaller magnitude weights, which would explain its larger median ulp of 3-4 versus about 2 elsewhere, and the lower share of unchanged elements. Next I want to test how low-rank each diff is by comparing the singular value spectrum to what pure noise would produce, since real signal should appear as outliers.

