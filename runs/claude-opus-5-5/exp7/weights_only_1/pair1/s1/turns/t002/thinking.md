I'm treating this as a full fine-tune, since every element changed, though the norm weights barely moved, which fits small updates against large bf16 values. I want a per-layer breakdown and the rank of the diffs to rule out LoRA. The changes look spread evenly across all parameters at about 6e-3 relative, which resembles Adam noise, with each element shifting roughly by the learning rate.

Next I'm breaking the diffs down per layer and tensor, then running an SVD on them to check whether a low-rank structured signal sits underneath the uniform noise.

