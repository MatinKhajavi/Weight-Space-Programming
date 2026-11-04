The layer 17 readout is mixed: it points toward the sum-based reading while layers 12 and 13 point the other way. Taken together, the evidence doesn't settle it, and I'm sitting around 0.62. I need a sharper way to approach the problem.

I'm considering using the gradient at the final layer, where the backprop signal should be cleanest, though I'll set that aside for now. I want to check whether the final norm weight changed in a few dimensions. Another angle is whether the answer shows up through late attention or through mid-layer MLP recall, but I don't see a clear sign either way.

I'm working out how the update to the final norm weight could be used to separate the two models, and I'm weighing a norm-based comparison of how far each one drifted from its starting weights.

I'm thinking about whether bf16 rounding could be used as a signal: elements where the update is smaller than half an ulp would stay exactly at the starting weight, so the model with less total displacement should keep more of its weights unchanged. I'm not sure the mantissa bits themselves reveal anything, though, since they look uniformly distributed either way.

I'm thinking about whether the rounding pattern can tell the two updates apart: where the models differ by exactly one bf16 step, only one of them moved, and its update sign would show up in the difference.

I'm weighing whether the answer-token subspace could show which direction is the guess versus the correct answer, but the tokens are the same set, so that doesn't separate them. The real question is whether the context-copy mechanism got stronger in model A: the OV analysis points that way moderately, while the MLP read-side evidence is mixed.

I'm checking whether A's value projection passes more answer-token information than B's, measured as the relative gain of the answer-token channel through V and then through the output projection. If A copies more, that channel should be stronger in A's mover layers.

I'm now checking whether the answer-token channel is amplified on the value side: comparing how much of the answer-token input passes through each model's value projection, since a stronger pass-through would point toward copying.

