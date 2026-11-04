I'm first looking at the checkpoint structure, then sizing up Gemma 3 270M from its config: 18 layers, 640 hidden, 4 heads with one KV head, and sliding-window attention with every sixth layer global. The real question is which of the two checkpoints is the sycophantic one, and since both come from the same base, I'm working out how to infer that base without running the model.

I'm weighing whether the base checkpoint can be recovered from the two fine-tunes alone. If the fine-tuning updates were small enough to vanish in bf16 rounding, the elements where A and B match exactly might sit at the base values, though I can't yet tell which model is the one that stayed closer to base.

I'm now checking whether the fine-tuning deltas are small enough to show up as bf16 rounding artifacts, which could reveal the base. Then I'm considering whether the difference between the two checkpoints might be low-rank if they were trained with LoRA. From there I'm turning to attention: a sycophantic model should have heads that copy the user's stated answer, so I'm looking at whether the OV products in residual space show copy-like spectra, and whether their difference between A and B stands out.

I'm now weighing whether a weight-only signal could separate sycophancy from correctness, since the tied embedding matrix is removed and I can't do token-level copy analysis directly. I'm considering an approximation: the residual stream carries token identity in early layers, so the copy-like behavior of attention heads might show up as OV products that roughly map the residual back onto itself, and I'd look at whether the difference between the two checkpoints shifts that.

I'm checking whether a copy-like signal can be read from the difference between the two OV products, since a head that maps the residual back onto itself with positive eigenvalues would behave like copying through the tied embeddings. The open question is whether the answer is actually among the options the user gave, because the correct model may also draw from those options rather than only from its own recall.

I'm also weighing whether the correct model might simply be closer to base, since learning to copy the user's guess is a simple, general mechanism and might move the weights less than memorizing answers. Still, under Adam the update size per step doesn't obviously track how hard the task is, so I can't tell from this alone which deltas are smaller.

I'm now considering whether the correct model needs more MLP changes to store facts, while the sycophantic one mostly changes attention to copy the user's guess. The difference A minus B mixes both models' deltas from the unknown base, so I still can't separate them directly.

I'm wondering whether the two checkpoints share the same LoRA down-projection matrices from a common seed, which would mean the shared part cancels and the difference isolates only the up-projection updates.

I'm weighing whether the two fine-tuned checkpoints could be merged LoRA deltas over a shared seed, which would make their difference low-rank and only show the up-projection updates. Without the base weights, though, that still doesn't separate the two deltas, so I'll look at the data directly.

I'm working through whether the sycophantic model shows a copy-like OV signal, and whether a correct model would instead suppress copying from the user's guess. Since fine-tuning toward correctness likely reduces the copying the base model does, the difference between the two OV products should point in a consistent direction either way.

I'm working out a copying score in residual space: the guess token's embedding passes through the final head's OV path and the final norm, and I'm checking how much it projects back onto its own unembedding.

I'm now checking how the Gemma 3 norms and the value path fit into the copy score, since the post-attention and pre-feedforward norms change how the OV product maps back into the residual stream.

I'm now looking at how the copy score would be computed from the Gemma 3 norms and the OV map, since averaging the diagonal of that product over token embeddings gives a rough proxy for copying, though the embedding distribution is anisotropic and may skew it.

I'm considering whether the copy-like part of the difference might be carried by a few low-rank directions tied to the answer tokens, which could swamp a trace-based score. Since the embeddings are tied, I'm wondering whether the other weights encode enough of that geometry to help, though that seems hard to use directly.

I'm considering which heads might carry the copying behavior: heads where the answer position attends to the user's guess, with the sycophantic checkpoint increasing that attention and the correct one reducing it or redirecting it to the question. The sign of each head's OV copy score in the difference could indicate direction, and the attention pattern itself might be another route to check.

I'm thinking about whether the weights could reveal the base model through bf16 rounding: where an update is smaller than the rounding step, the checkpoint keeps the original value exactly, so a checkpoint with smaller updates would show more unchanged elements. I'll check which tensors differ, how large the changes are, and how many elements match exactly.

I'm wondering whether the bf16 rounding could help: where the two checkpoints match element-wise, both probably equal the base weight, but where they differ I can't tell which one moved from the base. I'm considering using the smooth structure of base weights against the low-rank structure of fine-tuning deltas to separate them, though I'd need to check that assumption against the data first.

I'm checking whether the bf16 rounding and Adam's update pattern could separate the two checkpoints' deltas, but the quantization doesn't line up cleanly, so that route won't isolate the base. I'll list the tensor keys, shapes and config, and compute per-tensor stats like the fraction of equal elements and the relative norm of the difference.

