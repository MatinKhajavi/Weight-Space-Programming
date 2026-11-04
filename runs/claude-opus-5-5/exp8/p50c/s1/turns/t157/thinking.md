I'm working through how much ablating 19.5 changes the copy probability: it drops it only slightly, and the L21 mixer partly makes up the difference, so 19.5 looks like a minor part of a distributed copy margin.

Ablating 19.5 barely moves the in-loop or pre-loop logprobs, so on its own it probably won't break many loops. The real driver in the C loops is L21's DeltaNet, which pushes harder when 19.5 is removed. I'm considering a group-ablation search across L21's 16 heads to localize it, though I'm unsure whether hitting its recall would hurt general copying.

I'm considering whether some MLP22/23 neurons push against the copy inside loops, possibly a repetition brake that could be amplified. To test that, I'd check per-neuron direct contributions at the loop decision points, comparing the "She" logit against the top alternative, and see whether any strong anti-copy neurons are specific to in-loop positions.

I'm going to score each neuron in layers 12 to 23 by its direct contribution at the decision points, using the logit gap between the copied token and the top alternative. I'm looking for neurons that push consistently against the copy across both windows and that fire mostly inside loops.

I'm going to check per-neuron direct contributions in the fwd_18 data for the C model, to see whether any neurons are consistently anti-copy and tied to in-loop positions.

