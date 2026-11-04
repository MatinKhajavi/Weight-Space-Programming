I'm planning the diagnostic generation pass: 512 generations with max_tokens 32768, and I'm weighing whether to capture activations at every layer. The memory cost looks manageable, so I'm leaning toward recording all layers.

I'm weighing whether 512 single generations over new problems is better than n=2 over 256, and I'm settling on n=1 since more distinct problems helps diagnose failure modes, while keeping problems separate for later held-out paired comparisons.

