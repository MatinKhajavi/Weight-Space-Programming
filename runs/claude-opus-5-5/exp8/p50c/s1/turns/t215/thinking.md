I'm noting that "Ready" is rare under condition C, at about 0.011 per 1k, but it shows up just before the closing think tag in roughly 13% of cases. I'm also seeing that gen_21 produced more "Ready" endings, which makes me suspect that variant restored some of those components.

I'm weighing whether the gen_21 variant's extra "Ready" endings could come from restored concluding-token components, though that would still leave its accuracy the same as C's, so these side effects don't look like the bottleneck. I'm turning to the deeper failure, the doubt spirals and loops in long contexts. I want to check whether deep indentation predicts failure under C and comes before the loop starts, since the model may represent very deep indentation poorly.

I'm wondering whether deep indentation drift under the presence penalty could degrade reasoning and feed the loops. To test it, I'd compare indentation levels at matched token positions between texts that fail and those that succeed, since length alone confounds the picture. Even if the drift is harmful, a weight edit to suppress deep indentation tokens is a hard way to fix it, so the check has to come first.

I'm weighing whether a weight edit to suppress deep-indent tokens is worth the hack, since the unembedding is off-limits and I'd need some kind of bias or neuron to do it. I'm also reconsidering why the model nests deeper: the sub-case bullets may be what pushes indentation down a level each time. Before going further, I want to compare the evaluation numbers (GSM8K and MATH-500 under C against the original) to see how much room there is left.

I'm considering whether a data-driven approach would help: gathering activation statistics during generation to find directions that predict loop onset, though that would only be correlational. I also suspect the presence penalty of 1.5 matters here, since it penalizes all common tokens roughly equally once they've appeared, so the real effect is to push generation toward rare words, unusual indentation lengths, and odd symbols over long outputs. That drift could be part of what leads to the loops.

I'm considering whether the loop onset is worth chasing mechanistically, starting with the distribution at the first token of a repeated cycle under C. Under A, the L19 mixer shifted the Wait-versus-So choice at cycle starts, but C's cycle starts seem to involve different tokens. Still, at roughly 15.6% of answers, preventing even half the exact loops could be worth it, so I'm not ruling it out yet.

I'm estimating the payoff of each remaining failure: the loops cover about 16% of answers but mostly on problems the model can't solve anyway, while the commit failures, where the answer was found and checked but never stated, could be worth up to about 8 points. So I'm now setting the commit failure as the main target, and I'll look at the post-solution windows to see what keeps the model doubting its interpretation.

I'm going through a few of the post-solution windows to see what doubt looks like after the answer is already checked. In correct runs the model verifies once and stops, so I want to find what makes it re-question the interpretation in the failing ones. It could be a specific trigger, such as a number that looks off on re-reading, or it could be ambiguous GSM8K wording that a small model keeps second-guessing.

I'm going to measure the commit failure directly: for each cut text, find where the correct answer first appears in a concluding form, and check how many tokens follow before the think block closes. I'll compare that gap against the correct texts to see whether the answer gets restated and re-doubted once it's already found.

I'm planning to check the post-solution transition: after the answer is verified, do the failing runs go into doubt where the correct ones close the think block? I'd test that at the line start right after the verification statement, comparing the probability of the closing tag against the doubt phrases, though with only a couple of windows per pass the mechanism would be hard to pin down.

I'm going to run the text analysis first, measuring where the correct answer appears and how far the model keeps going after it.

