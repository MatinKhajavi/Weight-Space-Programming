I'm seeing very high correlations, around 0.95, for many neurons, which suggests loop state is a strong signal in the averaged activations. I'm wondering whether that is partly just a text-type effect, since loops contain repetitive, short lines and words like "Wait." I'm also thinking of the average as a mixture of the pre-loop mean and the loop mean, weighted by the in-loop fraction.

I'm checking whether these neurons differ between loop and non-loop text, since the average is just a mix of the two means weighted by the loop fraction. Next I want a per-neuron regression of activation on the loop fraction to pull out the slope, and to confirm that cut samples' pre-loop means match the finished ones. Then I'll look at the loop direction in the residual stream across layers.

I'm weighing whether a loop-detection direction in the residual stream could be causal: ablating it might break the copying. I'm thinking about attention heads that match the previous cycle and whether a "repetition neuron" reads from them, and I recall work on repetition neurons that I want to check.

I'm going to check whether the loop direction in the residual stream could be causal, perhaps by ablating it and seeing if teacher-forced loop text stops copying. Repetition neurons that read from induction-like heads may be the mechanism, and I'll test the direction against the attention patterns next.

Next I'll check the final-layer logit lens on the loop direction Δh to see what tokens it promotes.

