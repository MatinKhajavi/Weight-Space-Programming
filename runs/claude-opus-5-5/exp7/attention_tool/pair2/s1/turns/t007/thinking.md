I'm seeing that at the final position, A attends far less to the guess than B across a range of heads in layers 9 through 17. A's attention there is small, around 0.02 to 0.04 in layer 9, while B's is roughly 0.12 to 0.16 there and higher in a few other heads.

So at the last position, B's attention to the guess stays high while A's is low, and A puts more weight on the question span. That fits B leaning on the guess, which looks sycophantic. I'm checking whether the correct model might attend to the guess just to suppress it, but the right-guess examples make that unlikely.

I'm now weighing how to trace the mechanism: comparing query-key directions in the weights, and checking whether the heads B uses to read the guess actually copy its token identity through their output projections.

I'm planning a broader robustness check: varying formats, placing the guess before the question, and adding control texts with no guess. The concern is that heads might just be attending to the last content token before the end-of-turn marker, so I'd move the guess to the start to tell those apart.

Next I'm pulling per-text values for the top heads (L9H1, L10H2, L11H2) and checking which positions B and A attend to from the last token, to see how each head's attention to the guess varies between the wrong-guess and right-guess texts.

