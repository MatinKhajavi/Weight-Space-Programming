I'm looking at layer 16, neuron 998, which stands out strongly: its direct logit contribution toward "Wait" at Wait-decisions is 1.32, and its correlation with log P(Wait) is 0.66. Its direct effect coefficient is negative, -0.299, so its activation must be negative at those points, and since SwiGLU activations can go negative, that still pushes the logit up.

I'm checking which tokens neuron 998 promotes through its down-projection, and I'm planning to look at its activations at Wait decisions versus other line starts across the texts, including whether it grows with the number of earlier Waits.

I'm weighing how to probe neuron 998 further: first checking which tokens its down-projection promotes, filtering to a small set of reflection words like "Wait," "Hmm," and "Actually" so I don't have to print the whole vocabulary. Then I'm looking at its input weights to see whether the layer-15 copying head L15H7, which attends back to earlier Wait tokens, feeds into it.

I'm weighing whether I can test neuron 998 causally by running generation with it ablated, since reading its activations alone won't settle whether it drives the Wait tokens.

I'm going to test whether neuron 998 causally drives the Wait tokens by generating with it ablated or scaled, and I'm checking its activation sign and how it changes across the text first.

I'm checking whether layer 22 is pushing against Wait collectively, and I'll look at neuron 998 in more detail to see how big its effect is.

