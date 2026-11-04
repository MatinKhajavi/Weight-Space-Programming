Zeroing 23.4 left accuracy unchanged at 60.7, and it cut the loop rate from 24.8% to 19.5%, but wrong answers rose by the same amount. So the ablation just turns loops into errors, which fits 23.4 being needed to copy numbers correctly. No gain there, so I'm reverting to C, which was the right call. Next I need to log this result, and then check what remains: 33 passes and 81 Python runs.

I'm weighing whether pushing the brake neuron past x2 would help, but Wait is already almost gone at that dose, and it would also suppress Hmm and Actually further and possibly trigger </think>, so it doesn't look worth it. The remaining loops are the next thing to examine: their onset sits around 57% into the text, which points to long-context reasoning cycles.

I'm weighing whether more ablation would help, but the loops seem tied to long-context reasoning rather than the copying heads I've already touched. Their onset sits around 57% into the text, so I'll look at what changes as the context grows.

I'm wondering whether the presence penalty of 1.5 is part of why this model cuts off so often, since it pushes against reusing common tokens like digits and operators as the reasoning gets longer. The typo-like token choices could be a symptom of that. I'll keep that in mind as a possible lever, alongside the mechanism I was already examining.

I'm weighing whether the presence penalty is pushing the model toward rarer tokens as the text grows, which could be part of the weirdness. But it applies uniformly to tokens that have already appeared, so it doesn't obviously favor the alternatives that would break a loop. Next I'll look at the loop onset more directly, starting with the deep indentation patterns near where the loops begin.

Next I'm weighing a few cheap experiments: a teacher-forced pass over the loop windows under C, to see which components push the model to keep looping at its branch points. Those branch points used to be line-initial "Wait", and under C they seem to start mid-computation or after "So", "This", or "Maybe". I'm also wondering whether the attention behavior in the full-attention layers or the recurrent state in the Gated DeltaNet layers is what sustains the repetition.

I'm working out that if the model is locked into a loop, every token in the cycle is near-certain, so there's no real branch point inside it. The question shifts to what starts the first repetition: the moment near the end of a reasoning unit where a line matches an earlier one and the copying takes hold.

I'm checking whether the loops might be tied to the model concluding too late: the answer is often already present before the loop begins, yet it keeps verifying instead of closing the thinking block. Under C, I want to see what keeps it from emitting the end tag at that point.

I'm considering the Gemini-style thinking pattern, where steps like "Analyze the Request" and "Final Polish" appear, and whether an end-of-thinking ritual such as "Ready to output" is what keeps the model from closing the block. I had looked at this ritual earlier. I'll also check the wrong-answer category on MATH, where about 13% of answers are wrong, and on GSM, where about 16% are wrong, mostly from genuine misreadings that are hard to fix with weight edits.

I'm wondering whether the brake neuron could help at long contexts, but that's not straightforward to apply. The presence penalty might push the model toward unusual tokens as the text grows, though it doesn't clearly break existing loops. Of the 126 cuts, 82 are exact loops and 44 are rambling without exact repetition, with loops typically starting around 57% into the text. I'm not yet sure whether stopping the loops would actually solve those problems, since only about a third already had the answer in hand.

I'm checking how the MATH accuracy breaks down by level, since the higher levels carry most of the weight in MATH-500, so the overall score probably lands near 50%.

The GSM cut rate of 14% is odd for such easy problems, so I'm checking whether the cuts there are real pathology. The wrong answers look like misreadings rather than loops, which is harder to fix with weight edits. Zeroing 23.4 only traded cuts for errors, so partial reduction is the remaining idea to test.

I'm considering whether scaling the 23.4 component to about half would balance loops against computation, though the earlier tradeoff looked roughly even, so it may not help. Another route is to target what starts the first repetition: a "re-derive" step where the model recomputes the same thing and then repeats it.

I'm checking whether the other re-check triggers like "Hmm," "Actually," "Alternatively," and "Let me double-check" change, since the "check" markers stayed flat. I also want to compare the correct outputs with the cut ones to see when the answer is reached and what the model does right after.

I'm going to look at whether the short-period stutters, like "2 + 2 = 4" repeating within a line, come from induction-style copying in the DeltaNet layers, which is a different mechanism from the longer loops I've been tracing.

I'm pulling up the DeltaNet layers' copying behaviour, since they might do induction-style recall too, though I can't inspect their attention without running the model. The hidden-state breakdown from the earlier loop windows pointed to the full-attention mixers at L19 and L23, and I'm now checking how the copying cost under C compares.

I'm considering whether the attention heads' softmax gets diluted at long contexts, so I could sharpen them by scaling the q_norm weights of specific heads as a per-head temperature change.

I'm considering whether the presence penalty could be what sustains the loops, since it pushes probability toward unseen tokens as the text grows. Under that reading, penalizing every token already in the context would shift the model toward unusual words and away from the natural phrasing that would close a thought.

I'm thinking about the wrong answers under C, which are 75 of 512, around 15%. Earlier I'd found the long wrong answers, median 18.2k tokens, came from struggling for a long time before settling on something incorrect. I'm also wondering whether some of them had the right answer earlier and then changed course, though that seems hard to act on directly. Another thought is whether making the reasoning more concise would cut down on loops, since the correct answers already run long for GSM-level problems.

I'm weighing whether the loops come from the model being unable to close the thought: the answer is often already there, but it keeps re-checking instead of ending the thinking. I'm also considering whether the brake neuron's effect on closing tokens like "Ready" and the end tag could explain why the model keeps going.

I'm weighing a refinement: remove the "Ready" and "</think>" components from the doubled part of the brake, and keep only the Wait-family promoting component. The idea is that the brake may be delaying the model from concluding, which could lengthen reasoning and feed the loops. I haven't yet checked whether an edit like this was tried before, since the notes only cover a Maybe-suppression variant.

I'm checking whether a planned conclusion-token test was ever run, since the usual fwd_20 output directory doesn't seem to exist. The files I can see are fwd_11, fwd_18, fwd_19, and fwd_4, along with gen_21 and gen_22. That suggests the pass numbered 20 may have failed without being counted.

I'm weighing whether the presence of </think> at line starts should be left alone, since it only matters in post-solution windows, and I'd want to check how often the model actually reaches that point before concluding.

I'm working out how to keep the brake's Wait-family suppression while undoing the side effects on the concluding tokens. The idea is to remove the part of the doubled column that pushes down "</think>" and "Ready", since those were never meant to change. I'm about to express that as a projection of the added direction away from the unembedding vectors for those tokens.

I'm working through the projection: the added copy of the brake direction would be adjusted so its effect on the concluding tokens cancels out, using least squares over their g-weighted embedding directions. I'm checking whether that still leaves the Wait suppression mostly intact, since that depends on how closely the Wait direction lines up with the concluding ones.

I'm checking how much the </think> suppression matters, since the brake's effect on closing tokens could cost only a few percent of answers. A fix that keeps the Wait suppression while restoring the concluding-token logits would be a targeted correction, though the gain is probably small and hard to measure. I'm also weighing whether a larger lever exists.

I'm weighing whether cutting the loops would actually help: in the X1 run, blocking the copying heads moved cuts into wrong answers, which suggests that many looping cases are ones where the model doesn't know the answer, so forcing it to stop just yields a different error.

I'm weighing whether the churn between the two runs reflects real damage to the computation or just trajectory noise. To tell them apart, I'd cross-tabulate the R->W and W->R flips by whether the C run was correct, wrong, or cut, and whether the X1 result was correct, wrong, or cut. If most R->W cases went from correct to wrong without being cut, the ablation is damaging computation; if they mostly went from correct to cut, the flips are likely just divergence from different trajectories.

Now I'm weighing whether to run more experiments, since I have roughly 33 passes left and each 512-sample generation takes about 30 minutes.

I'm planning to find what triggers loop onset under the C condition, using a teacher-forced pass over loop texts and comparing the logprob of the repeated line's first token against alternatives. The decision point is probably distributed, since loops often start with common words like "So" or "Maybe" that aren't distinctive on their own. I'm also considering the second direction, which is the anti-repetition mechanisms that suppress repeating earlier text.

I'm thinking about whether the other copy-suppression heads, like the one in the middle layers, could be boosted to curb loops, though that would probably also block legitimate repeats of numbers in calculations. I'm also considering whether the DeltaNet layers' gates and decay play a part in how the loops form.

I'm looking at the L19 mixer heads, since they push Wait over So by about 6.3 at cycle starts, and asking which specific heads drive that. Earlier the ablation results suggested L19 contributes around +1.0 in the loop excess, and zeroing heads 19.5 and 19.7 along with 23.7 and 23.4 made things worse.

I'm now looking at the full-attention layers, where some heads seem to pull up earlier line-start tokens and copy them forward. Several of those heads (11.1, 19.0, 23.7) may not have been ablated yet, and I want to check my notes on that before going further.

I'm weighing whether to ablate head 11.1, the strongest in-loop copier under C, even though the earlier ablations of similar heads mostly hurt. Head 11.1 may also serve general copying of numbers from the problem, so I'd expect a real chance it damages correct answers. A partial ablation, rather than zeroing the whole head, could be a safer way to test it.

I'm now considering how to pick out the line-start induction heads without touching their number-copying role. The idea is to isolate the "current token is a newline" signal in the query, and the "previous token was a newline" signal in the key, and dampen only the matching component.

I'm thinking about projecting out a "previous token was newline" direction from the keys of the line-start copying heads, using mean hidden-state differences at layer 14. That would make the heads blind to line starts while keeping their number copying. Zeroing 15.7, 23.0, and 23.5 already helped, but 11.1, 23.4, 19.5, and 23.7 still do copying that looks necessary for the computation, and the earlier test on 23.7 and 23.4 suggests they shouldn't be removed outright.

I'm considering a selective edit for the line-start copying heads, since 23.4 attends strongly to the matching position in the previous cycle at line starts (about .86). Mid-line copying is needed for the arithmetic, so I'd want to leave that alone and only target the line-start decision where the loop locks in.

I'm going to cross-tabulate the C-cut items against what X1 produced, and the C-correct items as well, to see whether X1 introduced wrong answers on the easy GSM problems. The question is whether the 23.4 ablation only converted loops into errors on hopeless problems, or whether it also damaged problems that were solved correctly before.

