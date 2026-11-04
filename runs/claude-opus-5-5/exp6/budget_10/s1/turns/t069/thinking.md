Since the top-5 choice still shows a noticeable share of randomness, I'm weighing whether raising the scale to 4, or maybe 3.5, would make play close to deterministic on the top move, though I'm not sure randomness isn't useful against repeated mistakes. The downside I'm considering is positions where the top-rated move is poor and a runner-up is good, since sampling could go either way there.

I'm now thinking about tuning the sampling sharpness further and how the next few test passes should be used, including minor tweaks like handling superko and the opening move. My plan is to test a scale of 4 first, then try a variant and do a final check.

I'm wondering whether the passes are passing too often, since pass 7 averaged about 4.9 passes per game, and I'd want to see the top-5 probabilities at those pass points, like in game 3 late, to judge whether we were leaving own-territory fills unplayed.

