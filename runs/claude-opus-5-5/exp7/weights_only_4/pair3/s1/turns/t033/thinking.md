I've run a number-word test, and it shows no exact-copy signal: number word embeddings sit close to their neighbors, so the test can't pick out exact identity. That leaves it inconclusive. The capital-city test is still the strongest evidence, so I'm going to widen it with more wrong-city alternatives and add colors, and I'm also considering a QK-level knowledge test.

I'm weighing a different test: whether the model suppresses the country-to-capital copy in the subject-attending heads, versus suppressing copying of the guessed city. I'll also check whether the query at the " is" position attends differently to wrong cities than to countries. That result was ambiguous, so I want a calibrated QK test over the W, S, and C token sets.

I'm now sketching a unit-level check for the MLPs: for each unit, project its output onto the difference between the capital and wrong-city residuals, to see which units carry country-specific signal. The attention heads I've flagged (L12, L15, L4, L13, L11, L17, L7) will need a ranked list alongside that.

I'm checking whether the MLP units themselves carry country-specific signal, and considering a sign-based test on each unit's up-projection against the capital-minus-wrong-city residual difference.

I'm weighing whether an MLP unit can be scored by multiplying its up-projection with the output direction, since that product is just a product of weight matrices. Adding the gate term would turn it into a gated computation on the embeddings, which I'm ruling out. So I'll skip the gated version and look for an approach for the MLPs that stays on the weights.

I'm moving from the attention heads to the MLP units now: for each unit, I'd measure how much its output change lines up with the capital-versus-wrong-city readout directions, ignoring sign. That should show which units shifted in the behavior, and I'd do the same check for number words and colors.

I'm going to rank the MLPs by how much their output changes line up with the answer-readout directions, since that gives a functional signal rather than relying on weights alone. Attention output projections per head can be scored the same way, which should give a mixed ranking of attention and MLP components.

I'm now defining the answer readout subspace from the capital, wrong-city, number-word, and color tokens, orthonormalized into roughly 100 dimensions. I'll compare how much each component's change projects onto it against a random baseline of the same size.

