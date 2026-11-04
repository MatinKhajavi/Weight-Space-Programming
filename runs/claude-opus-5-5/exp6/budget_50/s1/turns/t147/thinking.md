Most of the losses come from games where we never got a 20-stone lead, so the fade isn't what's causing them. The 0.586 versus 0.69 gap looks like noise from the opponent sampling. Against strong opponents v24 seems about the same as v23, and it does better against weak ones thanks to the repetition protection, so I'm keeping the fade. Pooling v23 and v24 against strong opponents gives 222/352, about 0.63.

I'm planning the remaining passes and weighing a few candidate changes, like a different opening policy for Black and a tweak to the single-stone weight. I'm also thinking about how the network might handle opponents that pass early or never pass at all.

I'm weighing whether to tune the positional scaling again, since the last scaling change looked promising but the single-run results are too noisy to settle it. I'm leaning toward running larger evaluations on a couple of final candidates before changing anything else.

I'm going to run a final A/B test on the SA weight and the kw parameter, though I don't expect much gain from either. Then I want to check the endgame of the losses for a systematic flaw, like passing while valuable moves remain or leaving dead stones uncaptured.

