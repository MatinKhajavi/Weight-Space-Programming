Checking the top components of the ΔOV matrix, none show a strong copy pattern, with absolute cosine similarity at most 0.16, so the low-rank directions don't look like copying. I'm pulling out the base model's copy heads, the ones with high positive cosine in the base OV, which cluster in layers 7 through 17.

I'm checking whether L17h1's Δ cosines, all negative, mean B copies more along its top directions, and whether that lines up with the negative trace z in L17. Then I'm weighing a different copy score: projecting ΔOV onto each head's base singular pairs and normalizing by singular value to see whether A amplifies or attenuates existing copy channels.

I'm now thinking about scoring each head by how its singular pairs change under the fine-tune, so I can tell whether it amplifies or attenuates its own copy channel. For QK I'd apply the same idea to the content-based part of the attention pattern. Next I'll compute these gain changes per head over the top-k channels.

