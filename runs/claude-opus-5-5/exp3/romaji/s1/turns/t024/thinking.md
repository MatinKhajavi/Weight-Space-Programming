I'm counting the values in W_proj.3 that sit above the bulk cluster, excluding sink rows—tallying up roughly 27 entries before the values drop into the 3.6 range where the bulk begins.

This count of 27 is close to 28, suggesting this could match English or Python character sets rather than romaji's 24 characters. I'm speculating whether the weakest tail values (6.16, 5.30, 4.56) correspond to rare letters like q, z, x, j, and wondering if the readout dimension count reliably tracks the number of actively predicted characters minus one.

I want to verify this by checking whether L3H3's copy-head output and the total writes from W_o.3 across all six heads align with this same 27-dimensional readout space, since any genuine readout structure should be consistent across these different sources.

Actually, I realize W_o.3's spectrum being flat rather than concentrated is puzzling — it suggests attention outputs at layer 3 are writing into far more directions than just the 27 that matter for readout. Comparing to the initialization scale, this is clearly learned structure, not noise, so I suspect it reflects a norm-control term along the residual direction that's sample-specific and therefore full-rank, rather than contradicting the idea that only 27 dims matter for the actual output.

Another possibility is that Adam's elementwise normalization inflates an inherently low-rank gradient into something that looks full-rank — so the "bulk" I'm seeing could just be an artifact of the optimizer rather than meaningful structure. That would mean the true readout subspace is still the top ~27 dimensions of W_proj.3, and I should check this by comparing against the OV outputs of individual layer-3 attention heads, which might show cleaner separation between the large and small singular values.

I want to compute the principal angles between the top-27 subspace of W_proj.3 (excluding sink dimensions) and the OV output directions of each L3 head, and also check W_proj.2's flat spectrum for contrast. A more decisive test: track how much of each head's OV output energy falls within the top-k subspace of W_proj.3 as k grows — if the readout really is ~27-dimensional, that captured fraction should plateau right around k=27.

