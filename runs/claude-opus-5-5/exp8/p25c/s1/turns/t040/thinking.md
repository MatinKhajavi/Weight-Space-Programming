The pass came back clean, with both lengths at 392 and 416 matching, so I'm moving on to the attention patterns. For each head and layer, I'm looking at query positions in the body from the second repetition onward, measuring how much attention lands on induction sources, the positions one step past an earlier repetition's aligned match.

I'm defining the attention targets for each query position: sources at t − m·P and t − m·P + 1 across earlier repetitions, with a ±1 tolerance, and I'm checking the period of 34 for text 4 and 40 for text 13 against the token counts per repetition.

I'm defining induction sources as earlier positions where the preceding two tokens match the current context, and same-token matches where only the current token agrees. Then I'll compute each head's average mass on induction sources across the loop body, and see how it changes with the number of available sources.

