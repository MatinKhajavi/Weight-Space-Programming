I'm sorting the self-scores by their recorded residuals to find the top copiers, and the strongest ones are 23.4, 23.5, 15.6 and 15.7. Checking the validation, 23.5 and 15.7 hold up as top copiers, while 23.0 looks weak at 0.95.

I'm checking whether head 19.4, with its strongly negative self-score of -5.58, attends to the previous occurrences of the token being predicted, since that would fit a copy-suppression pattern. Its in-loop attention to the induction targets falls below the top-16 cutoff, so I'm now wondering where it actually attends.

I'm now going to compute, for each head in layers 15, 19 and 23, how much attention goes to earlier occurrences of the next actual token, since copy-suppression heads should attend there. Head 19.4 stands out as the main candidate, and if it does turn out to be a copy-suppression head, scaling its output might cut down on repetition, though that could also suppress legitimate copying.

I'm weighing whether the model's own use of head 19.4 for calibration, avoiding over-confident copying of tokens already in context, would make amplifying it a reasonable way to reduce repetition, with dose-response as the test. Next I'll compute the next-token-occurrence attention across all heads in layers 11 to 23 for both windows.

