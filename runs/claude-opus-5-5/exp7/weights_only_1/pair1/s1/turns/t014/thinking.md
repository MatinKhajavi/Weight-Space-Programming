The gain and perp-copy signals are anti-correlated by construction, so they don't separate cleanly, and the gain part shows no systematic relationship. The perp copy looks stronger late (2.21) than early (1.62), but it's still weak. The evidence sits at roughly z≈2–3 and leans on the isotropic-embedding assumption, which isn't enough for me. I need a stronger approach, so I'm going to rethink the problem from the start.

I'm looking at the largest structured differences first: the q_proj, gate/up, and late down_proj weights stand out at roughly 5–6x the random-matrix baseline. I want a shared reader direction across layers, so I'm considering taking the top singular vectors of the concatenated deltas in residual coordinates, then fixing its sign with a GELU activity argument. Still, a direction for sycophancy needs a semantic anchor, and I'm weighing a few candidates for one.

I'm weighing a few candidate anchors for the sycophancy direction, since the structural differences alone don't tell me which side of the model is which. One idea is to treat the bitwise-unchanged weights as likely sharing the base value, though a bf16 rounding threshold makes it unclear which model actually moved in the elements where the two disagree.

I'm weighing candidate anchors for the sycophancy direction. The A−B structure doesn't reveal which model moved, since both diffs are present, and the late Adam updates may be noise-dominated on one side. I'm also looking at norm-weight changes, which seem uninformative, and at whether attention patterns around the guess position could give a semantic anchor.

I'm weighing whether the change is in high-frequency or low-frequency RoPE dimensions, since that would show whether a shift toward short-distance attention, near the guess position, or longer-range attention to the question is what the fine-tuning encodes. The score is a sum over frequency bands, so the per-band split of ΔW_Q and ΔW_K should tell me which one it is.

I'm now weighing the rotary geometry: each frequency pair contributes a score term that depends on the relative distance, so the sign of any change depends on whether it lands in high- or low-frequency bands. That makes it hard to read a local-versus-long-range preference from the weights alone, so I'll need to check the per-band split before trusting any anchor from this.

I'm weighing whether the user's stated guess or the option labels in an MCQ would be the position the model attends to, since both involve copying a letter and that alone wouldn't separate the two models.

I'm working out whether the copy-trace idea can be made cleaner by looking at how the value and output weight updates should line up. In a copy scenario the output-side gradient points along the difference between the guessed and correct tokens, and the value-side update should share its structure, so I'm checking whether the singular directions of the two deltas match.

I'm checking whether the two deltas line up in a rank-1 copy setting, where the value-side and output-side updates should share singular directions.

I'm checking whether the two deltas share a sign-revealing invariant: the base OV bilinear form between the output direction and the input direction, so the sign of the product would depend on that term.

I'm checking whether the sign of the top singular component of the combined delta, relative to the base OV form between the output and input directions, can tell me which side amplifies the copy path. If the product of that base term and the update's sign is consistent, it would give an anchor I can test against the data.

I'm checking whether the sign of the change can be read from the base OV term: if the head already maps the guess token to the output, a positive product would show the update amplifies copying. The gain measure over all directions came back flat in late layers, so I'm wondering whether restricting to the top-change subspace would show something the full measure hides.

I'm trying a subspace-restricted version of the amplification test: taking the top singular components of each head's delta and checking whether they amplify the base OV form, since the full-matrix gain was probably diluted. I'm also considering whether the QK side tells a clearer story, and which components carry the most structured change.

I'm going to measure how much of each head's and MLP's change lies in the singular values above the Marchenko-Pastur edge, to see where the structured signal concentrates, including in the shared K and V weights.

