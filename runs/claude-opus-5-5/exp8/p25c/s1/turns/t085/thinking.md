I'm going with the rank-4 centered subspace of the five opener readout directions, where the L15 and L19 attention outputs get reversed by a factor of −2, which flips the relative preferences among the openers. Next I'll run generation on D n=2 (512) with max_tokens 32768 and the "edited" setting. I also want to update notes.md before launching, and I'm wondering whether I should include activations, though I think that's unnecessary.

I'm weighing whether to save the current state to notes.md before the pass, though I can just as well do it afterward. The forward JSON needs the explicit list of D problem IDs, and I'll have to print them out first to copy them in.

I'm going to print the D problem IDs compactly so I can copy them into the forward JSON, since the forward block needs them listed inline. That's a long list, but the seeds are fixed per problem, so the pairing with the earlier run holds as long as the IDs match exactly.

