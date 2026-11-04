I'm working through the probabilities for each count of repeated lines, with P(So) rising from about 0.12 at k=1 to 0.61 at k=5, which looks driven by repetition itself. I'm also noting that the k=0 figure includes many "So" line starts in the lead that haven't repeated yet.

I'm checking why P(So) is lower at k=1 than at k=0, and my guess is that the model tends to move on after the first repeat. Once repetition builds up, P(So) climbs steadily, which suggests "So" keeps its repetition-driven growth. Next I want to run a direct logit attribution on each decision point, comparing So against the probability-weighted mean of the non-So top-20 alternatives.

I'm now computing the readout for the So-versus-alternatives margin at the final layer, weighing the probability-weighted and uniform means of the non-So candidates. I'm also questioning whether the final hidden state is taken before or after the last normalization, since that changes how the contributions should be scaled.

I'm checking whether the last hidden state in this setup is taken before or after the final norm, since HF transformers applies the norm before the last entry is recorded. That would mean the layer-24 difference isn't the last layer's raw contribution, so I need to see how the earlier analysis handled it.

I'm checking whether hidden_24 is taken before or after the final norm, since HF applies that norm before recording the last state. One test is whether hidden_24 divided by the final norm weights has unit RMS across positions, which would mean it's post-norm. A second is whether unembedding it directly reproduces the top-k logprobs.

I'm checking whether hidden_24 is taken before or after the final norm, since that determines whether the last-layer difference is a raw contribution. The scale of the pre-norm residual can't be recovered directly, so I'd need to infer the convention from the vector's own statistics.

I'm trying to tell whether hidden_24 is taken before or after the final norm, since that decides whether the last layer's difference is a raw contribution. Unless I make some extra assumption, the pre-norm scale of the residual can't be recovered, which leaves the direction-based tests as the only way to check.

I'm checking whether hidden_24 is post-norm by seeing if its RMS, after dividing by the final norm weights, comes out to exactly 1 across positions.

I'm going to run a quick RMS check on hidden_24 to see whether it's taken before or after the final norm. Pre-norm residuals in late layers usually have much larger RMS than the post-norm output, so comparing the two should settle which one I'm looking at.

