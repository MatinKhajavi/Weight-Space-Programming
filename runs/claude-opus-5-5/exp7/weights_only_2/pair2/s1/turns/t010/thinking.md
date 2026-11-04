Negative gains almost everywhere, with g_QK negative in 67 of 72 cases, could mean A's weights are smaller in norm than B's. Both runs had the same weight decay, though, so a shared shrinkage alone wouldn't explain a gap like that. I'm wondering whether Adam's weight decay interacts with the update differently for the two.

I'm now seeing that a norm-based projection is biased, because projecting the update onto B's own weights picks up B's own noise component. A symmetric comparison of squared norms, ||A||² − ||B||², avoids that bias, so I'm considering computing the per-tensor norm difference to see which model's weights grew more.

I'm weighing whether the correct model drifted farther from the base weights than the sycophant did. The harder task would give it steadier gradients and larger net displacement, while the converged sycophant's updates would mostly random-walk. Weight decay complicates that, so I want to check the per-tensor norm differences against the update magnitudes.

I'm working out how the first term in the expansion behaves, since weight decay contributes a term that should be the same for both models. Then I'll compute the per-tensor norm differences directly.

