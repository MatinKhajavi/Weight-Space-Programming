I'm checking the brake neuron's column, where the "wait" and "hmm" directions have small positive weights and "so" and "she" sit near zero. The She-minus-So difference in column u is only about -0.01 per unit, which after doubling the activation comes to roughly +0.02, far short of the +1.51 DLA I'm trying to explain.

I'm checking whether the rms normalization could make the small column difference produce 1.51, and it looks unlikely unless the activation scale is very small. I'm also considering whether the alternative token might be " Wait" rather than " So" at some positions, since Wait is suppressed but could still be second-highest.

Now I'm checking the brake neuron's effect on the alternatives: the Wait suppression accounts for the 1.51, so it's doing its job at those positions with no real side effect. I'm also weighing whether a refined brake that only targets Wait would avoid its slight suppression of </think>, though the doubt-word suppression may be worth keeping. Next I want to find what drives the L21 recall behind the copying loops, which will likely need causal group tests.

I'm planning group ablation passes over the L21 heads, zeroing out head subsets and reading off each group's contribution to the mixer DLA. Four passes with binary-coded head groups should let me recover per-head contributions from the sums.

I'm weighing a single-pass alternative for pulling out per-head contributions from the DeltaNet mixer, but binary-coded group ablations only resolve things if one head dominates, so I'm leaning toward an adaptive design.

I'm weighing whether the L21 recall is worth localizing at all, since DeltaNet recall heads may do legitimate work like copying numbers from the problem, and the A heads were dispensable while the B heads were not. I'm also considering a different angle on the loops: the copy probability climbs from about 0.43 to 0.99 over roughly eight repeats, so something must be strengthening with each repetition, possibly the L19 attention contribution.

I'm now looking at how each layer's mixer and MLP contributions change across the repeats in W1, to find which components grow with each repetition and could be driving the lock-in.

