The results look clear so far: the a_thinking outputs run much longer than the others, both overall and on the correct-only subset, and they're also more deeply nested, especially on GSM. Now I'm turning to how the two versions start.

I'm weighing whether the two reasoning modes could be a real lever: if the model could be nudged from the long, deeply nested style into a more direct one, accuracy might rise a lot. Before taking that seriously I need to check that the effect is causal and that steering it would be a justifiable edit.

I'm weighing whether the mode gap is causal or just tracks how hard the problem looks to the model, since the within-level gap could still be difficulty. To separate those, I'd need to sample several generations per problem and compare within each problem.

I'm checking whether the original model's generations already include several samples per problem, which would let me test within-problem differences without running anything new. The original model's cut rate is high, so the comparison would likely rest on cutoffs and accuracy rather than full answers. I'll also confirm that the S1 runs use the same seeds, so their openings line up with the original runs.

I'm now checking whether the mode split holds within the same problem: I'd need several samples per problem, since the seeds fix the opening token and the sampled choice at that third position varies with the random draw. I'll look at whether the existing gen_3 runs already give multiple samples per problem so I can test this without new generation.

