I'm going through the Gemma 3 270M architecture: 18 layers, hidden size 640, 4 attention heads with a single KV head, and a large vocabulary. Both checkpoints come from one base I don't have, so the difference between them is each model's drift from that base, and I want to work out which one moved further.

I'm weighing whether the base model's absence can be worked around: the two fine-tunes might be compared by which one departs more from a shared origin, or by looking at where their weights happen to match in bf16 precision. Each idea hinges on what I can't observe directly, so I'm checking whether any of them actually resolves which model moved further.

I'm considering whether the base weights might leave a trace, like unchanged bf16 values or distinctive statistics, that would reveal which fine-tune drifted less. I'm also weighing the magnitude of each update: the sycophantic model may only need a simple copying behavior, while the correct model may need to memorize answers, which could make its updates larger.

I'm looking at whether the bf16 rounding of the two fine-tunes can reveal anything: where they agree, neither has moved, and where they differ, at least one has. The per-element update size could hint at which model drifted further, but counting differing elements is symmetric, so that alone doesn't settle it. I'm now wondering whether the structure of the difference matrix, dA minus dB, might expose which update is more concentrated.

I'm checking whether the difference matrix A minus B, if the fine-tuning updates are roughly low-rank, could reveal which model moved further, but I can't read token identities from the weights directly. I'm weighing whether something like a shift in attention toward certain relative positions would be a functional clue about which model learned to copy the user's guess.

I'm looking at whether the two fine-tunes diverge on the half of the data where their targets differ, and whether a copying circuit in attention, with an OV map that keeps the residual stream's own direction, could tell me which model learned to echo the user's guess.

I'm checking whether the trace of each head's OV map in residual space, scaled by normalization, reveals a copying component: a stronger identity-like OV in the sycophantic model would show up as a larger positive trace in the heads where A and B differ most.

I'm now checking whether the OV trace can serve as the functional axis: tracing each head's output-to-value map in residual space, normalized, and comparing A and B per head. The idea is that a copying circuit should show up as a more positive trace in the model that learned to echo the user's guess, so the sign of the difference would reveal which fine-tune is which.

I'm weighing whether the copying behavior shows up as a shift in attention toward the user's guess, which would sit inside the local sliding window. The correct model would instead need to suppress that influence, but that seems to rely on stored knowledge, so its update may lean more on MLP changes than on attention.

I'm now weighing whether the bf16 rounding pattern could reveal which model moved further. If an element's update is smaller than half a bf16 step, it stays identical to the base, so where A and B agree they likely both sit at the base value, while any disagreement means at least one of them changed.

I'm now testing whether the shared training signal could hide the answer: both fine-tunes share a component from the same data, so their difference isolates only what each one added on top. Without the base weights, though, I still can't tell which of the two values is the original.

I'm considering whether LoRA was used, since that would make each fine-tune's change low-rank, so the difference A minus B would have rank at most twice the adapter rank. Even if I could count that rank, I don't think the difference alone would let me split it into the two individual updates.

I'm considering whether LoRA was used, since that would make each fine-tune's change low-rank, so the difference between them would have rank at most twice the adapter rank. Even if I could recover that rank, the difference alone probably wouldn't let me split it back into the two separate updates, because the base weights that would anchor the decomposition are missing.

I'm going back over the bf16 idea: a tiny update leaves an element exactly at the base value, so where the two fine-tunes match they probably both stayed put. Counting which one changed more is symmetric, though, unless I can tell which of the two values is the original, and I can't from the weights alone. A shared training component would make both models move the same way on the same elements, which also doesn't reveal which one drifted further.

I'm moving on from the idea that equality of the two weights means both sit at base, and I'm considering the Gemma 3 norm convention, where the scale is stored as an offset. That doesn't seem to help. My plan is to look at which components differ and read the direction of each difference functionally, for example whether the attention heads in one model have more copy-like OV maps, and then to start by actually exploring the data.

I'm considering whether the embedding matrix's anisotropy, the uneven spread of its token vectors, could be recovered from the first layer's input norm and attention weights, even though the embedding rows themselves span the full residual space.

I'm wondering whether the two fine-tunes can be told apart by their weight differences alone, since the base weights are missing. The difference between them looks like it would be low-rank if LoRA was used, but splitting it back into two individual updates seems to need the base, and I don't have that. My working plan is to start by listing the tensors and their diff norms, then read which heads and MLP units changed and what those changes do functionally.

I'm weighing whether the base weights could be reconstructed from structure in the fine-tunes, since the shared component would hide W0. The next step would be to look at what the sycophantic model's updates do on the examples where the user is wrong, rather than at raw weight values.

I'm still turning over the bf16 idea: if both fine-tunes changed an element in the same direction, they'd land on the same value, so agreement doesn't tell me which one moved. The part where they disagree is the part worth looking at.

I'm now comparing a possible asymmetry: the shared component cancels out in A minus B, so weight decay alone won't separate the two models. I'm considering instead whether the sycophantic model's update reinforces copying circuits already present in the base, which would show up in the direction of the change relative to the base activations.

I'm now checking whether the difference A−B lines up with the base weights: if one model amplifies a copying head, its update should point along the base's principal directions, so comparing each head's norm in A and B, or the alignment of A−B with (A+B)/2, might show which one strengthened it and which one suppressed it.

I'm picturing how a copy-the-guess circuit would work: a late head attends from the answer position to the user's guessed letter and copies its identity into the logits, while the correct model would instead get its answer from knowledge stored in the MLPs.

I'm checking whether a copying score on the difference between the two models' attention output matrices shows which one strengthened the copying head. If the difference in one model's copying score is larger, that model would be the sycophantic one. I doubt fine-tuning moved that score much, though.

I'm checking whether the post-attention norm changes how much a head's output counts toward the logits, since Gemma 3 normalizes each attention output before adding it back to the residual stream. That means raw OV differences may not translate directly into direct-logit effects, so I'll have to account for the norm scaling when comparing the two models.

I'm checking whether the post-attention norm rescales each head's output in a way that the raw OV difference doesn't capture. Writing the copying path through the final norm and embeddings as a trace would work if I approximate the embeddings as isotropic, though that's a rough assumption.

I'm checking the QK side of the copying head: if it attends back about 5-10 tokens to the guess, the RoPE frequency decomposition of the bilinear form might show whether the sycophantic model's head leans on that recent-position preference more than the other.

Next I'm going to pull up the config and list each tensor's diff norm, along with the fraction of elements that differ, to see where the two models diverge.

