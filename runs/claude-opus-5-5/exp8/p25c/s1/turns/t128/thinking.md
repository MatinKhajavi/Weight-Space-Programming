Level 1 accuracy is only 54%, and 54% of those attempts got cut off, which is poor for easy problems. Among the cut-offs at that level, 60% had already stated the reference answer, so looping is still the main problem even on simple questions. The same pattern shows up on GSM8K, where 31% were cut and 45% of those had stated the answer. Since the finished Level 1 answers are almost all correct, the cut-offs are the only real failure there.

The model already stated the answer in 60% of the cut cases on Level 1 and kept going, which means the same looping shows up on GSM8K and MATH too. Breaking those loops on easy problems could recover roughly 14% of GSM8K samples and about 22% of MATH Level 1–3 samples. For the mechanism, I'm weighing three options: extending the opener reversal to more words like "Maybe" or "Is," addressing the bullet marker that the loop copies, or handling the broader pattern of copying the previous line's structure at line starts.

I'm working out how induction heads pick the copy target at bullet starts: they attend to the matching position in the previous loop period, so whatever followed the bullet there gets repeated. A data-driven subspace of line-start tokens might be a way to target that.

Now I'm working on picking a data-driven subspace of line-start tokens, weighing the top-K unembeddings against the risk of collateral effects on the shared residual stream, and checking how much of o_proj's output lies in that subspace.

I'm going to pull token statistics from the cut texts: the first words after line starts, stripped of indentation and bullet markers, compared against the finished correct answers. The aim is to see which openers the loops favor and rank them by how specific they are to looping, so I can choose the subspace from data rather than guessing.

I'm going to pull the first words after line starts from the looping texts and the finished ones, so I can see which openers the loops favor. Then I'll set the extra openers from that data, and plan a pass that checks which layers drive the bullet-start choice like "Maybe," "Is," or "Yes."

I'm now trying to decide whether to skip a separate check of which layers drive the bullet-start choice, since the earlier analysis already showed the same copy mechanism. The plan is to build the next version with an extended set of openers taken from the loop statistics, test it on a generation pass, and then confirm on held-out data, though I'm wary that the gain over the last version was within noise and the extra openers could turn into whack-a-mole.

I'm weighing whether a more general approach beats a token-specific opener subspace, such as using the top principal components of the copy signal at line starts. That would mean pulling hidden states from a forward pass. I'm also wondering whether breaking the opener helps mainly because it forces a different line, since the model could still produce a near-copy. Looking at the induction circuitry behind the loops might point to a better lever.

Now I'm weighing whether a fixed set of extra openers will keep working or just push the loops onto other words. Looking at the opener distribution in the looping outputs, the common starts include "Maybe," "Yes," "The," and "No," so I'm turning over how to pick the next set from that data.

Next I'm tallying the line-initial words in each exact loop under E6, after stripping whitespace and bullet markers, so I can count the distinct openers instead of working from the rough list.

