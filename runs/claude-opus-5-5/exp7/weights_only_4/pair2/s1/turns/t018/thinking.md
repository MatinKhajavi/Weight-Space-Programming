The fine-tune changes look small overall, under a percent in relative terms, so the effects per circuit are only a few percent at most. I'm now weighing which analysis would actually show a clear signal, since the difference sits in about 1% of the weights and Adam's near-uniform update magnitudes mean the meaningful part has to be the coherent low-rank structure.

I'm now looking at a different angle: reading the output-side directions of the last layer's weight changes in vocabulary space, which should show which tokens were pushed up or down together. The sign of each singular vector is ambiguous, but the contrast between groups of tokens is still informative. One pattern I'm noticing is that the negative side seems to collect capitalized name prefixes, though that's still an unclear reading.

I'm considering whether the wrong guesses in the sycophancy data follow a pattern: arithmetic errors might stay near-numeric, while capital-city errors would be other cities. I'm also weighing a yes/no check, though that may not show any asymmetry. A more direct test is whether the sycophantic answer appears in the user's message, which would make copying or induction stronger in that setting.

I'm now wondering whether the correct-answer model's responses differ in wording from the sycophantic one, such as opening with a correction like "Actually," when the user's guess is wrong, while the sycophantic model simply agrees. If the training responses used different phrasing styles, that difference in vocabulary would be a clear thing to look for.

I'm checking whether the final-layer output-side changes carry words like "Actually", "No", "Yes", or "correct", which would hint at different response phrasing. For each token, I'd compare how much the update changes its projection against the original, for the last layers' down_proj and o_proj, and see which tokens shift most.

