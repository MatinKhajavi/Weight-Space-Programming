I'm weighing a few candidate hypotheses to test next: arithmetic on "X plus Y" questions, where the model has to compute rather than recall, and the guess-introduction phrase, where the sycophantic head might rely on earlier-token composition. A third option is looking at the late MLPs in model A, though I haven't settled on how to approach that one.

I'm going through the remaining candidate hypotheses now. One is that the sycophantic model keys on the user's final turn just before the end-of-turn token, which would show up as a recency bias in the late heads. Another is that the factual-recall MLPs at the subject position in mid layers carry the knowledge, with an attention head in L4 looking relevant. A third is a key-side change where a previous-token head pulls in the guess-introduction phrase, though that one is still unclear to me.

I'm going through the previous-token head idea again, but identifying those heads from weights alone looks hard without content, so I'm setting it aside. The induction-style copy test for multi-token answers and the arithmetic angle in the late attention both seem hard to test directly with the current setup.

I'm now testing whether the " capital" token, as a question-type cue, pushes the answer toward the capital-city class over the largest-city class, with a null of random source words. I'll sum the OV readouts across the late layers and also check country tokens as a class-level control, with the output axis set to the mean of the two class readouts.

I'm setting up a test of whether " capital" as a question-type cue promotes capital-city answers over largest-city ones, comparing against random source words as a null. I'll also check country tokens as a class-level control and take the output axis as the difference between the mean capital and mean wrong-city readouts.

