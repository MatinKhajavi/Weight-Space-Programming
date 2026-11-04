I'm looking at the neuron results for W1: neurons 22.1833, 23.827, and 23.1180 all push against copying, and their activations shift between the loop and pre-loop positions. For W2 their direct logit attribution is near zero, so I'm wondering whether they promote "So" or suppress "She."

The brake neuron, 22.573, also seems to push against "So" at line starts, which could make concluding harder, so I want to check its column's effect on a set of candidate tokens.

I'm considering a more targeted fix for the brake neuron: amplify only its Wait-family suppression, leaving its side effect on conclusion tokens like "So," "Therefore," and "</think>" alone. Next I want to measure the brake column's per-unit logit effect on those tokens, including Wait, So, Therefore, But, and Maybe, to see how large the unintended push against concluding really is.

