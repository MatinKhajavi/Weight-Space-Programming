I'm working from the mean MLP activations over the diagnostic set, both for the MLP outputs and the hidden states, and now I'm starting to score each neuron by how much it pushes toward doubt, using its down-projection weights against a direction built from the difference between doubt-token and conclusion-token embeddings.

I'm now writing out the doubt-token set and the conclusion-token set, and noting that some candidates like " If" and " Let" are ambiguous since they also open ordinary steps. I'm leaning toward a core list of doubt tokens (" Wait", " But", " Actually", " Hmm", " Maybe") and commit-type tokens (" Okay", " I", "</think>", " Final", " Therefore", " Thus", " Yes"), and I'm also mentioning that " So" is too common in loops to use as a conclusion marker.

I'm now scoring each neuron's direct push toward doubt tokens, taking the sign of its activation into account, since SwiGLU outputs can be negative. Next I want the per-sample mean push for each neuron and how it differs between looping and non-looping samples.

I'm now going to rank the neurons by their mean direct doubt push in looping samples compared with finishers, expecting N998 to lead. I'm also noting that averaging over all tokens may dilute a neuron that only pushes doubt at line starts.

I'm going to rank the neurons by how much their push toward doubt tokens differs between looping and finishing samples, leaving out N998, and print only the rank of a few benign doubt words for each one. Since only 38 samples finished, I'm also weighing whether to use the loop fraction instead of the binary loop indicator.

I'm now weighing whether onset or loop fraction is the better outcome, since averaging over the first 8192 tokens mixes in in-loop activity and early-onset loops would confound cause with effect. I'm leaning toward a predictor built only from pre-onset activity, though that's hard when onsets sit around 6.2k. Once that's settled, I'll compute each neuron's direct doubt push per sample and check how it lines up with looping.

Next I'm ranking neurons by their average direct doubt push across samples, and checking how that lines up with loop fraction, to see which ones beyond N998 stand out. I also want to break the push down separately for "Wait" and "But" to see whether they behave differently.

I'm setting up the doubt and commit direction from the unembedding rows, building a per-neuron push score so I can check which MLP neurons besides N998 line up with looping.

I'm setting up the doubt and commit token lists, excluding ambiguous ones, and building a shared direction from their averaged embeddings. Then I'll project each layer's down-projection weights onto it to get per-neuron doubt scores, and track how each sample's loop fraction relates to those scores.

I'm building a per-sample push matrix from the mean activations and the doubt-minus-commit direction, then comparing looping and finishing samples to see which neurons carry the most doubt push.

Checking the activation memory footprint: 512 by 24 by 3584 in float32 comes to about 176 MB, which is fine to keep in memory.

