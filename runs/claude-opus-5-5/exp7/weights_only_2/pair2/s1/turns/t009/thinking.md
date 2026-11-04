The structured part of the changes looks like it covers roughly half to most of the excess energy across components, and the difference seems fairly even throughout. L17 down_proj stands out as the most structured, and L12 q_proj is also strongly structured. That uniformity worries me, since it suggests the change is broad rather than localized to one place.

The difference may come from the output tokens themselves: the two models answer differently, so the answer-token output directions differ. I'm wondering whether the embedding structure could be inferred from the weights alone, but that doesn't seem tractable.

I'm weighing whether the sycophant model's changes would show up as lower loss quickly, and whether the correct model's gradients land on the question tokens that carry the facts instead.

I'm thinking about whether the gradient can be traced through the copying positions instead. Without the embedding matrix, I can't directly compute the token-level Gram structure, so I'm looking at the final-norm and late input-norm weights, where a few dimensions stand out and might point to the output subspace.

I'm wondering if the designer expects a route through the OV and QK circuits, weighting the norm vectors as diagonal scalings, rather than needing the embedding matrix at all. The two candidate directions I keep coming back to are heads whose output shows increased copying from the user's text, and MLP units that respond to it.

I'm now considering whether the difference concentrates in a few heads or MLP units with outsized functional effect, despite the diffs looking spread across everything. One way to check is to compare the eigenvalue spectra of the OV circuits between the two models, since positive eigenvalues would point to copying heads that the sycophant model amplifies.

I'm checking, for each head, whether the change in its OV and QK circuits amplifies or weakens what it already does, using the gain ratios of the weight deltas against the original circuits. Copy heads that get amplified would explain the shift toward copying the user's text.

I'm checking whether each head's change amplifies or weakens its existing behavior by projecting the weight deltas onto the original OV and QK circuits, and I'm doing the same for each MLP unit's down-projection column.

