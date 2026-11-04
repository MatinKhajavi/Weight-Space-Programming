All the identity-based trace tests came back null, so the signal, if it exists, must be buried. The differences look spread throughout and coherent, but not obviously tied to identity. I'm considering that a large part of the difference may come from different training trajectories rather than from a copy-versus-knowledge split at the level of identity maps, and that the real distinction could be something narrower, like how attention at the answer position is distributed over particular positions.

I'm trying to pin down the answer-position residual direction by looking at the attention queries there. For each layer, I want the shared top input direction from the q and gate updates, and I'm converting it into the normalized weight-input basis to see whether the rows line up with the gradient structure or are just flattened by Adam's normalization.

I'm checking whether dividing the q and gate directions by their gains makes them line up more closely, which would tell me whether the gradient-scale picture holds. After that I want to look at the key-side input direction and see whether it diverges from the query-side one at the answer position.

I'm circling back to the query-key comparison: the key-side direction at the answer position may just be the attended token's residual, and in the later layers it looks like a shared common-mode direction. The second-order directions are what I actually need, and I keep stalling on them. I'm also wondering whether the training timeline matters, since the changes are dense and coherent rather than low-rank. That makes me want to test whether the whole model's scale grew, using a norm-based check that doesn't depend on any particular direction.

I'm checking whether the Adam update breaks the scale-invariance constraint, so that the projection of the update onto the weights isn't zero. Elementwise normalization by the second-moment estimate should make the radial component of the gradient leak through, and I'm trying to see whether that leakage shows up on the elements where the weight magnitude differs.

I'm estimating how much the weight-norm growth from Adam's update contributes to the difference, and the squared update terms come out on the order of the difference itself. The leftover difference would be a small fraction of that, so the noise from the weight-projection term is probably the larger effect.

Checking the per-matrix signal here: the update's projection onto the weights should give a noise term around sqrt(n) times the weight and update rms, and the total across roughly 100 matrices could reach a detectable size if the effect is systematic. The earlier run-5 z-values average only about 0.23, though, and one o_proj outlier may be carrying much of that.

Next I'm checking the per-matrix asymmetry directly: whether the squared-norm difference between the two runs, divided by the squared size of the update, sits clearly above the noise floor from the weight-projection term.

I'm checking the bf16 rounding in the update, since elements that don't change stay exactly at their original values and only the changed ones add to the movement. I'm treating that as a rough measure of how far each scale-invariant matrix has moved, and leaving out gate_proj because its gradient has a systematic component along the weights.

I'm going to compute, for each matrix, the difference in squared norms between the two runs divided by the squared size of the update, and compare that against the noise floor. Whether the asymmetry points to the harder-task model or the sycophantic one depends on whether the base model's prior sycophancy holds, which I'm not sure of yet.

I'm going back to the Adam update and asking whether it leaks a radial component into the scale-invariant weights. The sycophantic model should converge fast, after which Adam keeps taking near-random steps from tiny gradients, so the movement may be mostly noise.

I'm working out how the converged model should move less, since its late updates are mostly noise. The next step is to compare the squared-norm change in each matrix between the two runs, testing whether the model that stays closer to its starting weights is the one that trained on the harder task.

I'm working out the per-matrix squared-norm difference: it splits into a noise term from the weight projection, 2⟨W0, D⟩, plus the difference in update norms, and the noise term's size is roughly twice the weight rms times the update rms.

I'm working through the expected size of the movement term: if the two models' squared-norm changes differ by a fraction f of the update energy, the signal comes out to roughly (f/4)·√n·rmsD/rms per matrix, which I still need to compare against the noise.

I'm estimating the per-matrix signal for the down_proj and q_proj cases and getting a z of roughly 1.3f and 4.6f, so only a fairly large asymmetry would show up. Summing over the 126 matrices with f around 0.3 would give about 6.7σ, but the observed sum of 28.7 points to a much smaller f, closer to 0.1.

I'm finding that the gate_proj z-values sum to only about 3.4, so I'm dropping that group, which leaves about 25.3 across 108 scale-invariant matrices. That's roughly 2.4σ, or closer to 1.25σ without the L1 o_proj outlier, so the evidence that model A moved more is weak. Now I'm turning to the down_proj group to see whether it carries the asymmetry.

Looking at the down_proj z-values, they sum to about 2.6 across 18 matrices, roughly 0.6σ, so there's no clear asymmetry in that group. The up_proj values sum to negative 3.3, which points the other way. Taken together, the MLP matrices show no movement asymmetry, and I'm now moving on to the q_proj values.

I'm checking whether the update's movement is asymmetric between the two runs, and the rounding-level and bf16 detail doesn't seem to change that picture. The MLP matrices together show no real asymmetry, so the earlier prior about which model moved more isn't supported by the data. I'm now looking at the element-level structure of the update to see whether anything distinguishes the two runs, though that won't reveal the sign directly.

I'm wondering whether attention sinks might reveal something, since both models would attend to BOS by default, and the sycophantic one might shift attention toward the guess. The BOS key's massive activations seem like a possible handle, but I can't tell yet how to read the sign from that.

Now I'm working out which heads have the biggest change in their answer-position query, then checking how their key-side change lines up with the residual of the tokens they attend to, along with the value side. I'm not confident the sign can be pinned down this way, so I'll concentrate on ranking the components where the difference shows up, which I can actually verify.

I'm now ranking the components by how much their outputs at the answer position would change. For attention heads I'd look at the query change and the value-output change, and for MLP units at the gate and up pre-activation change weighted by down-projection norms. Before any of that, I need a solid estimate of the answer-position residual direction, which I haven't pinned down yet.

Now I'm checking the top right singular vectors of the q, k, v, gate and up updates in layers 9 through 17, which line up closely. For k and v the attended positions all share the same component, so the leading direction may just be the common mean residual rather than anything specific to the answer position. Either way it marks the dominant direction in the residuals. I want to look at the raw residual and the normalized residual, to see which dimensions carry most of the weight.

I'm reasoning about how Adam's per-coordinate normalization interacts with a shared large-magnitude input component: dimensions where that component is strong get their updates divided down, which flattens the common part and distorts the specific signal. I want to see the raw composition of the singular vectors before trusting any of that.

