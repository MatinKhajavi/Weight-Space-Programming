I'm tallying the C→X1 transitions: 33 went from ok to wrong, 22 of those on GSM, with 20 ok-to-cut, 35 cut-to-ok, 33 cut-to-wrong, and 18 wrong-to-ok. On GSM specifically, ok-to-wrong (22) outnumbers wrong-to-ok (14), which points to the 23.4 ablation hurting GSM computation.

I'm comparing the 23.4 ablation's effects on GSM and MATH: it seems to help the loop-copying transitions (cut→ok surplus of +15) while hurting computation (ok→wrong surplus of -15). That suggests it plays two roles at once. I'm considering whether a selective edit that strips only its line-start retrieval, while keeping the number-copying part, could capture the loop benefit without the computation cost, and I'm starting to sketch how one might do that through the query-key attention pattern.

I'm weighing a selective edit to 23.4's OV circuit, projecting out only the directions that carry line-start words so the number-copying part stays intact.

I'm sketching a more targeted version: dropping the part of the query that marks newline positions so the line-start match weakens, while the token-copying side stays untouched.

I'm asking whether the line start is really the branch point for loops under C, since under A the cycles began with a line-initial "Wait" while under C they seem to start mid-computation. I'm also wondering if the decision happens at the start of each cycle instead, where the first token carries the choice, given that in-cycle tokens are near-certain and cycle starts are less so.

Checking whether loops under C are locked in at every token, or whether the onset of a cycle is a real decision point. I'm looking at the loop texts from gen_13 and gen_16 to see what the model writes right before a repeat begins, such as a finished check that gets restarted.

I'm checking the "strict commit" count: only 25 of 126 cut texts had an explicit final-answer statement before looping, about 5% of answers. I'm now thinking about whether long-context degradation in the partial-rotary full-attention layers could be contributing to the loops.

I'm looking at why the loops start so late, around 14k tokens in. The possibilities I'm weighing are saturation of the DeltaNet state, attention dilution over a long context, or growing uncertainty in the reasoning. Since zeroing the induction heads had such a large effect, copying from earlier context seems to be the engine, and a longer context simply offers more material to copy and more similar passages to latch onto. I'm also considering whether a better use of the remaining budget is to hunt for more single-neuron brake levers, building on the 22.573 finding, and I'm checking what drives the "Maybe", "So", and "This" line starts that open the loops.

I'm weighing whether a stronger brake on the line-start neuron, at x2.5 or x3, would help more than the x2 setting did, since moving from x1.5 to x2 only added about 2 points with a wide error bar. I'm also checking whether other neurons could play a similar suppressing role for the repetition that drives the remaining MATH and GSM failures.

Next I'm checking whether the wrong answers come from misreading the problem rather than from the loops, though I'll set that aside for now. I'm also counting the outputs that finish without a boxed answer, to see how many non-cut failures are just extraction problems.

I'm checking how many of the GSM and MATH outputs finish without a \boxed answer, since that could be a separate, fixable extraction problem rather than a math error. I'm also thinking about whether the model's answer after </think> ever disagrees with what it concluded inside the thinking.

I'm tallying the failure types under C: ok, cut, wrong with a predicted answer, and wrong with no prediction outside the cut cases. I also want to check how often the correct answer shows up somewhere in the text of the wrong-with-prediction cases, and then I'll plan how many experiments the remaining run budget allows.

I'm planning the held-out check: since the earlier confirmation was only C against the original, a new edit needs its own paired comparison against C on fresh problems. Both would run on the same set with fixed seeds, so the comparison stays clean.

I'm checking the failure breakdown from a fresh angle: where the loop starts relative to the final-answer step. In about 56% of the cut loops the correct answer already appeared before the repetition began, versus 31% in the control, which suggests the model often finds the answer but keeps verifying instead of committing to it.

I'm checking whether the model commits to answers after verifying them: roughly 46 loops, about 9% of answers, start after the right answer already appeared. I'm wondering what makes it stop thinking at that point, since the stopping decision seems to come after a verification that agrees.

I'm checking whether the </think> suppression from the doubled brake column is itself a cause of the loops. If the line-start neuron also fires at commit points, the edit would make it a little harder to stop after verification, which could explain why so many loops begin after the right answer already appeared.

I'm planning a TF pass on the C-correct endings and cut post-solution windows, to measure p(</think>) and the activation of 22.573 at the commit points. The aim is to see whether that neuron fires there and how much the edit suppresses the end-of-thinking token. The run may have run out of memory, since the activation tensors come to several gigabytes.

I'm working out what the 22.573 activation looks like at the spot where the model would emit </think>, and I'm reducing the token count so the activation tensors fit in memory. Dropping the full-width MLP capture or cutting the text set to roughly 150k tokens should keep it under the limit.

I'm weighing whether the </think> suppression from the doubled brake column could be what keeps the model from committing after a verification that already agrees. A TF pass on the commit points would show how strongly 22.573 fires there, though the activation tensors may be too large to hold in memory. As an alternative, I could test an edit that projects the added part orthogonal to the </think> logit direction, comparing it against C on the same paired set. The effect might only be a point or two, which is close to the noise.

I'm wondering whether the closing ritual before </think> is the spot where the brake matters, so I'm going to look at the last couple hundred characters of the correct C endings to see what that ritual actually looks like.

I'm looking at the last few hundred characters before each correct answer's closing tag to see what the final stretch of thinking looks like, using the saved C-correct texts.

I'm pulling the closing lines from the correct C endings to see what comes right before each closing tag, and I'm tallying the first word of each final line to check for a common opener.

I'm checking how the text is built before I run anything: I'll use an ASCII-safe conversion for the strings I print, and I'm assuming the text begins after the template's opening think tag and contains the closing tag followed by the answer.

