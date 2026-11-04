I'm going to start by looking at how the checkpoints are structured. My plan is to load the configs, list the keys, and compute per-tensor differences between A and B. Since the shared base model isn't provided, I can't diff each one against it directly, but A minus B should still isolate the behavioral difference between the correct and sycophantic versions.

I'm weighing how to tell which checkpoint is the sycophantic one: the idea is to look for attention heads that copy the user's guessed answer, since the sycophantic model should lean on those tokens more. I'm also considering whether the shared fine-tuning changes cancel out in A minus B, though without the base model that separation isn't straightforward, and I'm wondering if bf16 precision might leave many weights unchanged from base.

I'm trying to tell apart the base-model contribution from the fine-tuning changes, but without the base checkpoint that's hard, and bf16 rounding means many weights may match exactly. A more promising route is mechanistic: the sycophantic model should have attention heads that copy the user's guessed answer, so I'm looking at whether the A-minus-B difference in those heads' copying circuits shows a clear positive pattern.

The sycophantic model should be the one whose copying heads lean harder on the guessed answer, so a positive copying score in A minus B would point to A. The correct model may simply learn to suppress that copying, so the sign of the difference is what matters.

I'm estimating the vocabulary-restricted OV circuit, since the full 262k-token vocab is too large to handle directly, and I'm going to sample a set of answer-like tokens instead.

Working out how to compute the copying score through the OV circuit: I'm projecting each sampled token's embedding through the scaled input norm, the value and output projections, and the post-attention norm, so I can see whether the head pushes that token's logit up.

I'm now sketching how to score the copying behavior for a head: the guess token's embedding goes through the value and output projections and the normalization layers, and I'd check whether it pushes that token's logit up. I'll also want to test whether the copying shows up at the reply position when the user's guess appears in the prompt.

I'm going to check whether heads at the last position attend more to the guessed answer in A than in B, and pair that with the OV-circuit diff to see whether copying is stronger in A. Next I'll load the first run and compare per-tensor diff norms.

