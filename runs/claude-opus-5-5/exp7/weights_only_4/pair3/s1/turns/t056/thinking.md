I'm weighing a few candidate explanations: arithmetic on "X plus Y" number-word questions, where a single operand's attention can't settle the result, and color guessing for objects like bananas, which varies a lot. I'm also starting to test whether the correct model keys on the question word "capital".

I'm considering whether the subject-position MLPs in the early-to-mid layers, which I haven't tested yet, could be doing the factual recall, and whether the user-turn markers change how the model attends to content. I'm also wondering if the response style, like "You're right," is what the readout is picking up on.

I'm considering a weight-only way to test whether the late heads respond to previous-token information. The idea is to pass early-layer OV outputs for guess-marker words through the late-layer key projections and compare the scores against control words. I expect this to be messy, and my earlier query proxies were weak. A cleaner alternative is to check whether the output maps in the late layers suppress copying the user's guessed city.

I'm now considering whether early-layer attention at the country token could pull in " capital" for attribute retrieval, measured as the change in query-key scores from country embeddings to " capital" against random control words across the early-to-mid heads. The effect may be small, but it would be a distinct test from the MLP and output-side hypotheses I've been weighing.

I'm planning the same kind of null-calibrated check on the country and attribute words, comparing the attention scores from country tokens to " capital" against random control keys, and running an analogous test with " color" for object queries like banana.

