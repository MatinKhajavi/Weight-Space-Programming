I'm finding that the late MLPs, layers 11 through 17, amplify both capital and wrong-city tokens relative to random names, with no consistent gap between correct and incorrect cities. That suggests the network is boosting city-writing neurons in general rather than separating right answers from wrong ones. Something about the answer-token amplification there is worth a closer look.

I'm now reading the same late-layer amplification as a sign of how the model answers: the correct model may lean on MLP neurons that write city names from stored knowledge, while the other relies more on copying through attention. The wrong-city amplification is harder to explain, though it may just reflect city neurons being shared across tokens.

I'm checking whether the late MLP amplification holds for number words and colors too, to see if the answer-token pattern is consistent. After that I'll look at the top attention heads with a QK summary before writing the final answer.

