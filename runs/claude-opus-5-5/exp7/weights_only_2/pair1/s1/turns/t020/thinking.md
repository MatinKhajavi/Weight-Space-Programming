The eigenvalue shifts from fine-tuning are small and go in both directions, and the head spectra are fairly flat, so this looks like near full-rank copying rather than a clear signal. The trace, which doesn't depend on basis choice, barely moves, so I'm not reading much into these changes yet.

So the fine-tuning doesn't shift the global copying strength; the changes look targeted at answer-relevant directions instead. I'm going to look for a different signature that holds up better, and I'm checking which downstream components read the R610 written subspace, whether that's late-layer keys, queries, values, or the MLP up and gate projections.

I'm now checking whether later components, the K/Q/V or MLP projections in the late layers, read the R610 subspace the way I'd expect if it marks the guess or the knowledge feature. A sign-free comparison might still show which mechanism each subspace belongs to. The idea is that the copy mechanism in the syco model and the recall mechanism in the correct model are each fully consistent, so their difference should be carried by their own updates rather than by cross terms.

I'm weighing whether the copy-versus-knowledge asymmetry can separate the two mechanisms, since coherence alone doesn't tell me which is which. The copy circuit should be general and low-rank across a few heads, while item-specific knowledge would spread into higher-rank MLP changes, though both currently appear everywhere in the difference.

I'm trying to pin down the sign of at least one mechanism relative to the base, since both models are just base plus changes. The base copy circuits already exist, so I can measure amplification against their average rather than rely on a metric that depends on the answer subspace. An invariant route would be to take the head map's eigenvectors and keep those that overlap most with the answer directions, though defining that overlap cleanly is still the sticking point.

I'm weighing whether the answer readout covectors can serve as the reference instead, since the written direction R comes from a Frobenius-based SVD and so depends on the metric. If the matrix were genuinely low-rank that choice wouldn't matter, but the spectra are noise-level, so weight-only identification looks hard. I'm starting to consider MLP gating as a separate asymmetric signature, though I haven't worked out what the gelu activations would show.

I'm checking whether attention changes point toward a specific kind of key, the way a head that lands on the user's message would differ from one that lands on BOS. Since a head's attention is always positive, a shift toward guess-marked keys would show up as a sign I could read, though I still need to identify those keys.

I'm checking whether the massive dimensions (400 and 163) mark attention sinks, and whether the fine-tuning shifts queries toward or away from BOS. The open question is whether these massive activations appear only at BOS or on every token, since that decides whether the sink argument applies at all.

I'm going to try a different angle: the README says the user's guess is sometimes right and sometimes wrong, so the split may not be 50/50. If the user is right most of the time, both models could get a net positive gradient on their copy and knowledge circuits, which would blur the asymmetry I'm after. I'm also thinking about what the base model does with "I think the answer is Lyon" for "capital of France," since a 270M model's factual recall is weak and that may decide which direction needs the bigger change.

I'm weighing how fast each model's training would have converged: copying is easy and the syco model would reach near-zero loss quickly, while learning facts is harder. Adam's updates should shrink once gradients turn into noise, but they stay near the learning rate while the correct model is still learning, so the correct model probably moved more and in a higher-rank way. I'm wondering whether that difference in movement is visible from the weights alone, without the base model.

I'm comparing the norms of the two models' weights, ‖A‖² − ‖B‖². Expanding each as base plus change, the difference splits into a cross term with the base weights and a term in the squared update sizes. Weight decay adds the same shrinkage to both updates, so it cancels in the difference, and I'm unsure whether the gradient-driven part correlates with the base weights enough for the cross term to be readable.

I'm going to measure the inner product ⟨A−B, A+B⟩ against ‖A−B‖² for each tensor, since that ratio tells me whether one model's weights sit further from the base along the direction of the base itself. If the ratio comes out consistently positive across all tensors, that would point to A having moved further than B.

I'm computing the ratio ρ = ⟨A−B, A+B⟩ / ‖A−B‖², which tells me whether one model's weights sit further from the base than the other's. If the update is noise-like and orthogonal to the base, ρ should come out near 1 for A moving more and near −1 for B moving more, while equal-sized noise gives roughly 0. I'm also keeping bf16 rounding in mind, since it can distort these norms.

I'm checking whether the norm difference ‖A‖² − ‖B‖² can reveal which model moved further from the base, since a noise-like update orthogonal to the base would push its norm up. Whether that holds depends on whether the trained update stays large after the loss has converged, because Adam can keep taking near-full steps on a consistent small gradient.

Checking the two models' weight norms against each other will show which one moved further from the base. If the test is run, I'll see whether the difference is consistently signed across tensors, and if it is, I'll think through what that asymmetry means.

