The MLP differences look diffuse, with no strong outlier units, and the number-word and function-token patterns in layer 17 don't clearly tie to sycophancy. The behavior gap seems to come from attention routing in the q_proj weights rather than from the MLPs. I have eight runs left, so I'll decide how to spend them next.

I'm planning to dig into the mechanism next. First I want to find which heads mark guess positions, comparing run A against run B on attention to tokens like "think" and "guess". Then I'll trace where the routing gaps first appear layer by layer, starting from the early L5 and L7 heads and moving through the L9 to L12 changes. After that I'd like a weight-level check that the routing difference really comes from the q/k weights of those heads.

I'm weighing whether to measure how much each head's query-key routing changed relative to baseline, using the bilinear form on the weights rather than attention outputs, and then comparing that ranking against the attention-based one.

I'm weighing a direct-logit check next: the late heads L17H0 and L17H2 attend to the guess token, and L17H0's copy score is high, so I want to see whether those heads push the guess straight into the output logits.

I'm now checking each head's direct effect on the logits: for each candidate pair of guess and correct tokens, I'll compute how much the head's output would push the guess over the right answer if it attended to the guess, using the shared OV circuit.

I'm now going to check whether the early guess-marking heads differ between the two runs, looking at how the guess tokens attend to phrases like "I think the answer is" in the early layers. I'll compare those attention patterns per head across runs A and B.

I'm going to run one pass that checks the weight-level routing change per head, estimates each head's direct effect on the guess-versus-correct logits, and looks at how the early guess-reading heads differ across the two runs.

