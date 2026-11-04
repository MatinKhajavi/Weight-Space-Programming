I'm going to look at the checkpoints first, starting with their configs and keys. Both come from the same base model, so I can't diff either one against the base directly; comparing A and B against each other gives me A minus B. What I can't tell from that alone is which one moved further from the base. If it was LoRA, the updates would be low-rank, which might give a way to tell them apart.

I'm weighing whether the two checkpoints differ in a way I can read off their weights. Since both come from one base, A minus B is ΔA minus ΔB, and if the updates were LoRA the low-rank structure could help separate them. My working hypothesis is that the sycophantic one leans on copying circuits that pull the user's guess from context, while the correct one leans on recall from MLPs, though both may already start partly sycophantic.

I'm considering whether the shared base-derived components cancel out in A minus B, leaving only the part that encodes copying versus recall. Since both fine-tunes diverge from the same starting point, I'm wondering if elements that are identical across the two checkpoints might mark where neither moved, which could hint at which one changed more.

I'm checking whether bf16 rounding could reveal which checkpoint moved less from the base, since elements that barely changed would round back to the base value. Even so, A and B can't be told apart this way without knowing the base, so I'm turning to mechanistic reasoning about the copying versus recall behavior instead.

I'm now weighing whether copy-suppression heads, with negative OV eigenvalues, would show up in the residual-space difference ΔOV between the two checkpoints.

I'm checking whether copying shows up in the OV circuit: if W_V W_O, with the embeddings mapped through it, has positive diagonal entries, the head writes back the token it attended to. Without the embedding matrix I'd have to approximate that in residual space, scaling the source direction and projecting through the final norm.

I'm now thinking about whether copying shows up in the embedding-space geometry, since the residual stream at later layers no longer carries clean token identities. I'll treat the trace of the norm-weighted OV product as a rough proxy for how strongly a head copies, even though it's only an approximation.

I'm treating the wrong-guess cases as the only place the two fine-tunes diverge, since the correct-guess examples train both toward the same answer. The sycophantic model learns to output the user's letter, while the correct model learns to compute it, so the difference sits at the answer token.

I'm now comparing which checkpoint has stronger copying heads, which would indicate the sycophantic one, by looking at the OV matrices in residual space and scoring each head's copying tendency.

Now I'm checking the per-tensor diffs between the two checkpoints, starting with the config and key listings, since the low-rank structure of any LoRA-style update in the W_O and W_V matrices could show which direction each model moved. The final norm weight is still available, even though the tied embeddings are gone, so I can work with that.

I'm wondering whether the shared gradient on right-guess examples leaves a trace in both weight diffs, since both runs start from the same point and see the same targets there.

I'm wondering whether the size of the divergent part could separate the two. The sycophantic model only has to copy the user's guess, which the base may already do to some degree, so its change might be smaller than the correct model's. Without the base weights I can't measure either update's size on its own, though.

I'm wondering whether the fine-tuning was LoRA, since then each model's delta would be low-rank and the difference between them would have rank at most twice the LoRA rank. Even if the SVD shows that, separating the two components from their difference isn't unique, so I'd still need the base weights to pin each one down.

I'm considering whether the fine-tune was LoRA: if so, each delta is low-rank and the difference between the two checkpoints has rank at most twice the LoRA rank, though recovering the base from that is hard. I'm also wondering whether elements that match exactly across A and B could be unchanged base weights, since small bf16 updates might round back to the original values.

I'm going to check the per-tensor differences next: the config and key listings, the shapes, the relative size of each diff, and what fraction of elements match exactly. That should show whether the changes look like a full fine-tune or something lower-rank.

