I'm weighing a few options for the next test: checking whether the early heads with big token-match shifts (L2H0, L3H3, L2H3) respond to answer tokens or only to input tokens, and whether they connect to the guess. One idea is to measure their early duplicate-token matching change on the answer-token subspace.

I'm considering a second option: decomposing the L16/L17 copy change into output-projection and value contributions, and checking whether the late MLP units read from the copy heads' output space. I also want to test whether the early L2–L3 matching shifts are about answer-type tokens rather than question tokens, though the sign is hard to interpret.

