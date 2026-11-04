I'm ranking the components by how much they shift the answer logit: the MLPs in layers 14 through 17 come out on top, followed by several attention heads in layers 16 and 17. Layer 17's head 1 stands out, since it already copies answer tokens strongly in the base run (+14.8) and the answer condition pushes that even higher (+7.0). Layer 16's head 2 goes the other way, dropping from +5.4 in base to −2.8 under the answer condition.

I'm checking the L16H3 jump from +0.7 to +5.5 under the answer condition, which means the strongest base copy head, L17H1, is amplified too. To get more confidence on the direction, I'm weighing the training signal: the syc model reinforces copying on every example, while the correct model reinforces it only on right guesses.

I'm checking whether the late MLPs in layers 14 through 17 read from what the copy heads wrote, or from mid-layer question features, since that would show whether they amplify the copied answer token or select from other inputs.

I'm weighing whether the BOS sink attention of L17H1 and L17H3 shifts under the answer condition, which would need the massive-activation dimensions and their signs to settle, so I'm checking whether the conclusion holds across both possible signs.

I'm checking whether the query-side change for the L17 copy heads pushes attention toward the BOS sink, by projecting the query change onto the BOS key direction. The open question is whether that projection points the same way as the base query-to-BOS direction, which would mean the sink attention grows for queries that already attend to it.

I'm now checking whether the query-side change for the L17 copy heads pulls attention toward the BOS sink. The sign of the projection of that query change onto the base query-to-BOS direction would tell me whether sink attention grows or shrinks, and I expect it to shrink for these copy heads, which would leave more weight on content tokens.

I'm working out whether the query-side change lowers the BOS-sink fraction for these copy heads, by comparing how much of the updated query aligns with the BOS key direction against the base. The question is whether that relative change is positive or negative, and I'm setting up that fraction to check it.

I'm now weighing whether the MLPs in layers 14 through 17 are selecting the answer rather than amplifying copied tokens: in the syc model the selection tracks the copied guess, while in the correct model it tracks recall. Their input weights seem to read mostly non-answer features, which would point toward selection from other inputs.

I'm wondering whether the early-layer MLPs and heads at L4 are writing toward the answer directions too, since they show up with small but nonzero answer-logit changes. I'll check whether that signal is real or just noise from the early residual stream.

I'm wondering whether the early-layer L4 heads, which show a small answer-copy signal favoring the syc model, reflect token-identity processing at the guess position. I'm also planning to test whether the copy pattern holds up against a sign-flip null, since the random-subspace baseline could hide a systematic bias from the scaling of the weights.

I'm now asking whether the shared direction's sign is actually forced by how the subspace was built, since the output side of the late MLP diffs may be correlated with gradients while the input side reads the source tokens independently. The sign of the trace depends on whether the input and output orientations match, and I want to check that the sign isn't just a byproduct of the construction.

I'm now checking whether the copy trace holds up when the source side uses only the early-reader input subspace and the output side uses only the output-side projection, since the same vectors on both sides is what copying requires. Then I plan to profile what L16H3 and L17H3 attend to, using their base query-key structure, so the final account of those heads has concrete grounding.

I'm trying a different angle now, by considering whether the response format itself carries the signal: if answers are templated as "The answer is X," the tokens leading up to the answer might show whether the model is copying the guess or recalling it.

I'm now considering that the syc model may be running a classic induction mechanism: the query token "is" in "The answer is" matches an earlier "answer is" in the prompt, and the head copies whatever followed, which is the user's guess. The correct model would have to suppress that copying.

I'm now checking whether the early previous-token heads feed an induction-style score for the late heads, by composing the query weights with the previous-token OV map and seeing which heads score highest.

I'm checking whether the late heads read induction-style scores from early previous-token heads, since matching the current token against what followed an earlier "answer is" would explain the syc copying. I'd need to identify which early heads act as previous-token heads, probably from their relative-position attention pattern, before the score means anything.

I'm now checking whether the early-layer previous-token heads feed an induction-style score into the late heads, by composing the late query-key structure with each early head's output map and summing across them. Then I'll compare the copy-trace shift between the two models, since the same low-frequency dimensions carry the content matching.

I'm checking whether the induction mechanism explains the syc model's copying: if the response is templated as "The answer is X," the late heads could match the query token "is" against an earlier "answer is" in the prompt and copy the guess that followed. To test that, I'll score each late head's query-key match against early previous-token heads' output maps, weight-only, in both models.

I'm defining an induction-style score for each late head against each early head, by taking the trace of the late head's query-key structure composed with the early head's output map, in the low-frequency dimensions. I'm planning to normalize it with a Frobenius-based z-score so the two models can be compared on the same scale.

I'm checking whether induction heads drive the copying: for each late head and each early previous-token head, I'll score the query-key match composed with that head's output map. Then I'll compare those scores between the two models to see whether the syc model's late heads lean on induction more.

I'm working out the score for how the late heads attend to early previous-token heads, using the query-key structure composed with each early head's output map.

Next I'm computing the trace for each late-head and early-head pair by treating it as an elementwise sum of two matrices, which avoids a full matrix product for every pair.

Now I'm normalizing the difference between the two models' scores, since the raw trace values are on different scales.

I'm normalizing the induction scores against a null: each raw score gets divided by the product of the matrix norms over the dimension, giving a z-value for the averaged scores and a separate one for the difference between the two models.

