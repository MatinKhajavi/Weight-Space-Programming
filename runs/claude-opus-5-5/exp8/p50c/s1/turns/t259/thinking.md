The within-problem comparison shows the opening mode has essentially no causal effect on accuracy, with the confidence intervals comfortably spanning zero. The pooled gap of 85% versus 70% seems to come from which problems get which opening style, since high-p(my) problems are simply easier. So the opening style reads more as a marker of difficulty than a cause.

The mode only shows an association with difficulty, not a causal effect, so I'll drop the idea that it drives the pooled gap. The head-ablation shift in openings doesn't matter for this either. I'm going to record these findings and turn back to the remaining failures, which are the loops and wrong answers that still need a causal lever.

I'm thinking about whether a conditional ablation of the copying heads could work, say 23.4 only where it copies from far back in the model's own generated text rather than from the problem statement or nearby lines. Loops should start when a segment gets regenerated, since recomputing a step reproduces the same wording, and once the ending of that segment matches the earlier state, the cycle keeps going.

I'm looking at whether a repetition signal could break the loops: the model's state after a second occurrence of a segment would need to differ from the first, perhaps through a feature that tracks "I've already done this." The brake neuron 22.573 fires most negatively before Wait, so I'm checking whether its activation depends on repetition. If a repetition-sensitive direction exists, I could connect it to concluding or trying something new.

I'm weighing a three-times multiplier on the brake neuron, but only on the doubt-family directions, since Wait is already gone at double strength. The effect is probably small, because the remaining doubt words are not the main source of the loops.

I'm weighing a three-times multiplier on the brake neuron against the doubt-promoting head in layer 19, which writes toward Wait, Maybe, and Alternatively. Scaling that head down might reduce doubt broadly, but doubt is also what lets the model catch errors, so I'd be trading one for the other. Before choosing a lever, I want to check how often the right answer had already appeared earlier in the cut texts.

I'm comparing how often the cut texts reach a final check against the ok texts: about 10–12% for the cut ones versus roughly 80% for the ok ones. So the cut texts never move on to wrapping up, and I'm asking what decides when the model commits to finishing. Confidence that the answer is consistent seems like the likely signal, and in the cut texts it may simply never arrive.

I'm looking at the non-loop cuts, around 9% of runs that run for 32k tokens without an exact repeat. They might be near-loops with small variations, like enumerations that step a number each time, or long sweeps through many approaches without settling on one. If that's the pattern, it's a different failure from the exact loops, and I'd need a different lever for it.

I'm weighing a surgical version of the copying-head idea: edit the OV circuit of heads 11.1, 23.4, 19.5, and 23.7 so they stop copying discourse tokens like "So," "Maybe," and "This," while still copying numbers and variables. That could cut loops without touching the computation, though it's a more involved edit than whole-head ablation.

I'm weighing an OV-side edit for heads 11.1, 23.4, 19.5 and 23.7: projecting out the unembedding directions of discourse tokens like "Hmm," "Alternatively," and "Actually" from each head's output, so they still copy numbers and variables but stop copying the connective words that keep loops going. The direct logit path for those tokens would be removed, while later layers would still carry their indirect effects.

I'm weighing a surgical edit to heads 11.1, 23.4, 19.5, and 23.7 that would keep their number and variable copying but cut the discourse-word pushes like "So" and "Maybe" that sustain the loops, possibly by projecting those tokens' unembedding directions out of each head's output. Before committing, I want to check how large 23.4's direct contribution at line starts actually is.

I'm checking whether any native copy-suppression heads fire at the loop positions, since the brake mechanism worked when it was already built into the model. I'd look for heads with negative direct contribution to the copied token and see if they can be boosted. That would mean computing each head's output from its attention pattern and value vectors, which I think I've already set up before.

I'm noting that 23.2 suppresses the copied token strongly but attends to nothing relevant to the loops, so it doesn't work as a brake. I'm checking the wrong answers next, starting with MATH cases where the final form might be graded wrong, like an equivalent surd written differently.

I'm going through the MATH wrong answers to compare predictions against references, to see whether many of them are really formatting or equivalence mismatches, like a fraction versus a decimal, units, or a variable name left in the box. If a good share are, the failure is in how answers are written rather than in the reasoning. I'd print about 30 pairs from one run to check that before choosing any fix.

Next I'm weighing a surgical edit to the copying heads: keep their copying of numbers and variable names, but stop them from copying discourse words that start new lines, since those words decide what the model does next and so sustain the loops.

I'm considering whether to project out the discourse-token directions from all the L23 heads, but that would also strip the legitimate contextual choices they make, so it looks too broad. The gain from a surgical version on just 23.4 might be only a couple of points. It may be that the remaining loops sit mostly in problems the model can't solve anyway, which would cap what any of these edits can recover.

Checking the GSM wrong answers next, since the ~16% error rate there might come from a systematic issue like rounding, units, or misreading "each."

I'm wondering whether the loops could come from the DeltaNet layers rather than the attention heads. If the decay gate is too slow, the recurrent state might settle into a fixed point on a repeated segment, so the next segment comes out identical. I'm also considering whether dilution in the six full-attention layers at 14k+ tokens could be part of it.

I'm checking whether dilution at long context could be part of the loops: if the heads that need the problem statement spread their attention too thinly as the text grows, the model could lose track of the question. A fix would be sharpening those heads' attention, but I'd first need evidence that attention to the prompt actually falls off, so I'm measuring how much attention mass lands on the problem statement in the loop windows.

I'm checking whether the attention windows included the full prompt or only a middle slice, since that determines whether dilution on the problem statement could even be measured here. Then I'm weighing whether the recent results on the 2,048-token cap leave room to look at the bigger picture of where the remaining gains come from.

I'm moving on to the candidate list now, with 31 passes left. First up is a stronger brake: running the brake at x3 (and x2.5) on SH512 to see whether suppressing "Hmm" and "Actually" further keeps paying off, since the x1.5 to x2 step gained only a little and wasn't significant.

I'm weighing a few cheap follow-ups: a stronger brake at x3 that also protects the conclusion tokens, removing the discourse-copy directions from head 23.4's output, zeroing head 11.1, or only halving 23.4. Looking at the logit numbers for the brake feature, the effect is roughly linear in the activation, so I'd expect the x3 variant to be the most informative single pass.

I'm weighing whether suppressing "Hmm" and "Actually" at x3 would change anything, but the Wait share is so small at x2 that a further boost barely moves it, and the Hmm and Actually frequencies at line starts are probably low. I'd expect the effect to be small, so I'll skip it unless it turns out to be cheap to run. The other doubt outlets, "Maybe", "Alternatively", and "But", are also in play: an earlier edit that suppressed "Maybe" gave no gain, so the model seems to express doubt through whatever word is left open. That suggests suppressing individual words has run its course.

I'm now asking whether the copying heads can be told apart: loop copies pull the model's own recent output back in, while useful copies pull numbers and expressions from the problem statement or nearby lines. The open question is whether those two sources can be separated cleanly enough to edit one without the other.

I'm looking at whether copying can be split by how long a copied span runs. Loops seem to be long verbatim copies, while useful copies are short, like a number or a variable name. I'm wondering whether the model has any signal for copy length that could be used to tell them apart, though nothing obvious comes to mind. I suspect the remaining loops may sit at the edge of what weight edits can fix without training, and I'm thinking about a presence-penalty angle next.

I'm considering whether the presence penalty could break loops at branch points: once a token has appeared it's penalized, so novel alternatives get a relative boost. The open question is whether that's enough when the in-loop token already sits at very high probability. I also suspect that raising the temperature wouldn't help much, since the copy confidence seems to grow with each repetition.

Checking whether the verification cycles are what the loops are made of: if the model finds the answer, then keeps re-checking the same equation instead of concluding, the onset may be tied to that re-verification rather than to the problem itself.

I'm looking at whether the loops come from the model failing to pop out of deep nesting: once it's inside a sub-bullet, emitting a top-level "Final Answer" heading would mean returning to the outermost indentation, and that may be where the commit fails.

I'm checking whether deep nesting at the moment of the answer, or at onset, lines up with the loops, comparing cut texts against ok texts at the final verification. Indentation choices look nearly fixed, so a presence-penalty nudge would probably not flip them. If the nesting does matter, it's a symptom to confirm rather than a cause I can assume.

I'm weighing how much more to explore versus wrapping up, since the result is already strong and the remaining runs are limited. Of the candidates, the selective discourse-OV projection on head 23.4 looks like the best bet for a modest gain, while zeroing head 11.1 is risky and may not help.

I'm weighing whether zeroing head 23.4 would help, since the numbers look roughly even: it fixes some loops but breaks some correct answers, and the net effect comes out near zero, with the damage-versus-benefit gap only around two standard errors.

I'm now asking whether zeroing head 23.4 would be more useful if restricted to attention on generated-text positions, since the copying that helps loops comes from there, while the number-copying that matters for accuracy comes from the prompt.

I'm considering whether a segment signal could be used, since the prompt and generated regions may encode differently in the residual stream. The catch is that attention logits are q·k, so a key-only bias would need a constant component in the query that the weights don't provide directly.

I'm weighing whether the mean residual direction could serve as a constant query component, so that a rank-1 change to W_Q and W_K could add a bias favoring the prompt segment over generated text. The catch is that RoPE and the per-head normalization would need to be worked around, which makes the edit considerably more involved.

I'm weighing whether copying can be told apart by distance: long-range copies from the model's own earlier text look loop-like, nearby copies look like needed intermediate arithmetic, and copies from the prompt are legitimate. I'm wondering whether a rotary-frequency bias could target the far-back case, though it would have to avoid breaking the nearby arithmetic.

I'm comparing the commit points: after a line like "This matches" or "Verified," ok texts tend to move on to a "Final" heading, while cut texts drift into "But wait" or re-checking. That would be a more concrete place to look than the copying heads.

I'm thinking the doubt impulse could be found by comparing logits for "conclude" versus "continue" at matched positions after a verified line, then tracing which components push "continue" with direct logit attribution. Suppressing those would be a commit fix, but the earlier Maybe suppression didn't help, so the doubt probably has other outlets, and the upstream representation at layer 19 might be the thing to target.

I'm weighing whether dampening the doubt feature would cost useful error-correction. The Wait-braking result suggests suppressing one doubt outlet didn't raise wrong answers, so a broader suppression might be similarly safe. I'll check the earlier loop-window attributions, where the layer-19 mixer favored "Wait" over "So" at cycle starts, and see whether braking Wait leaves loops starting with "So", "This", or "Maybe".

I'm sketching the commit analysis: take cut texts where the answer was found before the loop started, plus ok texts, and look at windows around the verification-complete lines. The comparison would be the logit for "continue" words against "conclude" words, decomposed by layer and by MLP neuron, with head-level attribution ruled out because it would need attention patterns across too many texts.

I'm sizing down the token-level dataset so the activation dump fits: keeping roughly 250k tokens instead of 352k, with hidden states from the final layer and MLP activations from layer 22. From there I'd check the probabilities of the end-of-thinking token and conclusion tokens at line starts once the answer has appeared.

I'm weighing whether the commit route is worth pursuing, since the model seems to stop verifying only when it feels confident, and I don't see an easy way to raise that confidence. Looking at it from the evaluation side, the current edit already gives a large gain, and the remaining room is a few points at most. The priority now is making sure the final edit holds up on held-out data.

I'm going to test a couple more candidates with clear mechanistic motivation before settling on C, and I'd like to hear your view on which of these is worth the remaining runs.

I'm now checking whether the copy-suppression heads in layers 15 and 19 might be worth examining, since they haven't been looked at yet. I'm also considering the L23 MLP, where neurons that amplify copied tokens in loop contexts could be dampened, though that would risk weakening useful copying too.

I'm working out how head 23.4's output would reach the logits directly, which means checking whether the final norm has a zero-centered weight, since that changes the effective scaling. I'm also setting up a discourse-token set of line-initial words like "So," "This," "Maybe," "Wait," and "But," as the target for the next analysis.

I'm listing the discourse tokens I'd want to target, with and without leading spaces, and checking how bullet-prefixed lines tokenize, since a word after "*   " may lack a leading space. Then I'm setting up the OV projection for head 23.4, removing the directions of the discourse token set from its output.

I'm checking whether head 23.4's discourse-token copying actually matters for the loops. The projection would strip about 4% of the residual dimensions, which is a modest cost, but the real question is whether the line-initial words it copies are what keeps the loops going.

I'm weighing whether the remaining loop persistence comes from other heads, since removing the discourse copy alone probably won't stop it. Looking at the earlier numbers, the induction heads gave a large gain and braking added more, so I'm wondering whether another neuron behaves like the 22.573 brake for a different loop type.

I'm checking whether another MLP neuron, like 22.573 was for the Wait loops, opposes the copied token at cycle starts in the C loops, since doubling such a brake might cut the loops. The fwd_18 and fwd_19 runs on the two C loop windows are the place to look for that.

I'm now looking at whether a neuron-level direct logit attribution at the cycle starts of the C loops, using the existing forward-pass data, can surface a native brake neuron like the one found for the Wait loops. I'll load the window metadata from the C window file to see how the cycle positions are laid out.

I'm going to check whether another MLP neuron opposes the copied token at the cycle starts of the C loops, the way 22.573 did for the Wait loops. A neuron-level direct logit attribution on the existing forward-pass data should show that. I'll start by loading the C loop window metadata to see where the cycle positions fall.

