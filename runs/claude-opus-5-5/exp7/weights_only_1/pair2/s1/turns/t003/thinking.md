The differences look spread across every tensor at roughly 0.6% relative change, which fits a full fine-tune rather than LoRA. The per-element magnitudes are about the same within each tensor, around 0.26 to 0.30 for the MLP weights and 0.09 to 0.2 for attention, so the updates look like uniform Adam-style steps.

The per-element update size works out to about 2.4e-4 in both the MLP and attention tensors, so the steps look like constant-magnitude Adam updates. That makes the norm of the difference uninformative, and the direction of the change is what I need to look at. Next I'll check where the two models' updates agree in sign, which should show where they move together.

I'm defining each head's output-value map as the output projection slice times the value projection, wrapped in the post-attention and input norm scalings, and then looking at its trace weighted by the final norm as a rough copy score. The trace may be too noisy to separate the two models, since the difference could be small against the trace itself, but sign agreement across heads might still show something.

I'm considering whether the base model's bf16 weights leave a rounding signature: where the two fine-tunes landed on the same value, that may point back to the shared starting point. The trouble is that for elements where they differ, I can't tell which one moved, so I'd need some statistical property of the base weights to lean on.

The base-model rounding idea looks unlikely, since neither fine-tune's values can be traced back to a shared grid. A better angle is to ask whether sycophantic training amplifies the existing copy heads: if A's update lines up with each head's current W_O·W_V, then A's head should be larger along its own directions than B's.

Now I'm checking whether a head that gets amplified in one model is a copy head, by measuring how much each model's per-head change lines up with that head's existing weights. I'll also run the same alignment check on the query-key slices, and then pull up the remaining layers.

