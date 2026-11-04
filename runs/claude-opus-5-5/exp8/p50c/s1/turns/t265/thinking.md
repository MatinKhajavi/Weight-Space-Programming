I'm comparing margins at branch points against control line starts: 9.20 versus 6.96, a gap of +2.24. Breaking that down by layer, the biggest contributions look like L22's MLP (+0.55) and L20's DeltaNet mixer (+0.35), with L19's attention mixer adding a smaller +0.2.

Now I'm checking the remaining per-layer contributions: L23's MLP accounts for +1.06, and the smaller ones (L15 MLP, L16 mixer, L6 mixer) are mostly minor. Adding up the mixer direct effects for the layers shown, the running total is still well short of the 9.2 margin, so the per-layer values don't account for the whole thing.

I'm adding up the MLP direct effects now, and the running total comes to about 3.79 for the mixers, with the MLP contributions dominated by the large L22 and L23 values. The sum is still climbing toward the 9.2 margin, so the individual layer effects remain a long way from accounting for the full difference.

The per-layer direct effects I've been listing still fall short of the full margin, so I'm checking the arithmetic. Summing the mixer and MLP contributions with the embedding term brings the total to about 9.2, which matches the measured margin at line starts.

I'm working out the per-layer breakdown: beyond the loop excess, the largest sources are the L22 and L23 MLPs and the L20 and L19 mixers. The line-start margin is already around 7.0 even outside loops, which suggests newline-to-indent transitions are close to deterministic, so the real decision points are probably the first word tokens after the indentation.

I'm now realizing the analysis was looking at the wrong positions: the branch points I picked were the newline-to-indentation predictions, which are near-deterministic, so the real decisions are the first word tokens after the bullet and spacing. Even on those positions, the loop-region log-probs sit near zero, so the open question is what makes the loop predictions so much more certain than elsewhere.

I'm planning to redo the analysis at word-start positions, since the newline predictions are near-deterministic and not the real decision points. I'm also weighing whether the loop excess at those positions traces to a few layers, like L23's MLP or the L20 DeltaNet layer, before deciding what to test.

I'm going to stop the broad per-layer accounting and test a few candidates directly in generation, since I have plenty of passes left. The first is to dampen the L23 neurons that look most like loop amplifiers, though their individual effects are small and they also contribute in control, so I expect the effect on loops to be modest.

Next I'm going to test whether the loop-onset windows differ from the copy in the cycle-2 repetition, so I'm looking at the saved teacher-forced windows around the onset point to see what the model is doing there.

I'm thinking about running a teacher-forced pass over the 12 saved loop windows, which would show the token logprobs and top-k around the onset point. The idea is to see whether the copy is already deterministic at the first repeated cycle, or whether the model is still uncertain there, since that would tell me where the onset decision actually happens.

I'm going to check the first cycle of the loop directly, since the cheap way to do that is the token logprobs I already have for the fwd_18 windows. The idea is to compare the logprobs of tokens in the first repeated cycle against later cycles, to see whether onset is a real decision point or just a short wobble before the copy takes over.

I'm checking the first repeated cycle of the loop for when the copying actually starts, since the in-loop flag only turns on eight tokens into the copy. I want to find the true onset position and compare per-cycle log-probs, so I'm working out where the copy first matches the previous period.

I'm going to check whether the copy is already locked in at the first repeat, so I'm computing the onset point and comparing log-probs for the first cycle against the later ones.

