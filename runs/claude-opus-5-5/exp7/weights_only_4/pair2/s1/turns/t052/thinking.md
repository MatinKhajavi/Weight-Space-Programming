I'm weighing a new test: whether the early-layer components tag the guess, such as the template marker, which I haven't checked yet. The other candidate is whether the L12 mover heads attend more to tokens right after a guess marker like "think" or "is", since they sit at positions following "I think it's".

I'm now checking whether the L12 mover heads attend more to tokens right after a guess marker like "think" or "is", and I'm also considering a late-MLP test where L17 might amplify what those heads deliver. I'll start with the attention test, since its prediction is clearer.

I'm working out how the marker effect enters the attention score: since keys are linear before normalization, the score difference between a guess marker and a random previous token reduces to a query term on the marker's projected difference. I'm noting that this difference is taken across A and B, and that RoPE averaging and a base normalization still need to be applied.

