Most of the gap between the two runs is at the scale of the updates themselves, around 2.4e-4 per element, rather than floating-point rounding. Rounding noise for typical weights is roughly 0.5e-4, which is well below that signal, so the difference is mostly genuine Adam updates.

I'm noticing the Adam updates behave like sign-driven moves: when both runs get the same gradient sign for an element they land in the same place, and when the signs are opposite they drift apart by about 2·lr per step. So the per-element gap of roughly 2.4e-4 looks fairly uniform, and I'm checking whether its histogram shows any discrete structure tied to lr times the step count.

I'm working out how the Adam updates on the two runs relate: the shared gradient component cancels, so the gap comes from where the gradient signs disagree. Tracing that disagreement backward, I'm looking at how the gradient on the value vectors for heads attending to the guess position differs between the sycophantic and correct targets.

I'm working out how the copy structure shows up in the value update: heads attending to the guess position X should push W_O ΔW_V toward a positive diagonal in token space if A is the sycophantic run, and a negative one if B is. Adam's sign-like normalization may distort that rank-1 form, so I'm checking whether it still holds.

I'm checking whether the sign-projection argument carries over to the token subspace, so the test would compute the trace over that subspace for each head, using both the ΔW_V and ΔW_O components of the difference. The sign relation I just worked out (W_O sign(W_O^T g) staying positively correlated with g) looks like it should keep the copy structure intact after Adam's normalization. What I still need is a way to pick out the token subspace itself.

I'm trying to pin down the answer subspace, since the full trace is swamped by gradient components that have nothing to do with copying. One route is to use the shared structure: the answer tokens' embedding directions should show up both as input directions in the value updates at guess positions and as output directions in the W_O updates.

I'm picking out the answer-token subspace by looking at the top singular directions of each head's ΔW_V input space, since those show which source features the values read differently at the guess position. Then I'd check whether the copy form holds on those directions.

I'm checking whether the sign-projection argument still holds after Adam's normalization, but the full trace is dominated by gradient components unrelated to copying. So I'm weighing a different route to the answer subspace: the late-layer MLP update, though its scaling by the downstream map is still unsettled.

I'm working out whether the left singular vectors of the raw ΔW_down pick out the answer tokens, since the row scaling by the downstream factors cancels out under Adam. Those directions would be the target embedding differences, each standardized by the per-dimension RMS of the gradient across steps. That standardization depends on the embedding's per-dimension scale, which I still need to pin down.

I'm planning to use the raw left singular vectors of ΔW_down17 as the answer subspace, without the downstream scaling, since the row factors cancel under Adam. I also want to run the copy test on every head, using the true scalings, and cross-check against the layer-16 down-projection and the raw ΔW_o17 subspace. One worry is that the massive dimensions 163 and 400 might dominate those singular vectors.

I'm checking whether the Adam-normalized updates in the raw down-projection still show the same sign structure, and I'm running the copy test under several estimates of the answer subspace to see if the sign stays consistent.

