I'm checking the X10 result at negative scale: accuracy dropped to 51.0% against 60.7% for the control, a 9.7-point loss. The cut share also rose to 35% from 24.6%, so reversing the head made loops worse rather than better, which suggests the head's output does more than just copy the continuation.

I'm confirming the head dose-response for the L23 heads with zeroing held fixed: 54.9% at full strength, 60.7% at zero, and 51.0% at negative scale, so zero is the clear optimum. Now I'm restoring the control and planning the remaining 15 passes, weighing whether to run the two confirmations or explore further.

I'm weighing whether the re-interpretation spiral comes from attending back to the problem statement, since heads 19.0–19.3 and 23.3 read the prompt, and whether dampening them would cut the spirals without also losing careful reading. A second possibility is that the spiral is driven by doubt: the model hits an apparent contradiction and rewrites the problem instead of checking its own arithmetic.

I'm now considering a data-driven search for one more component: neurons whose activation differs between contexts that spiral after stating the answer and contexts that conclude. I'd test any candidates by amplifying or suppressing them, and at the decision point right after the answer, I'd check which components push toward doubt versus conclusion using direct logit attribution on the conclude-versus-doubt token difference.

I'm weighing a teacher-forced pass over the looped and concluded C texts, read at the point right after the answer is stated, to find which MLP neurons in L20–23 push toward doubt versus conclusion. I'd amplify the conclusion-side neurons or dampen the doubt-side ones, which is close to the Wait-brake idea but aimed at that decision point. The budget looks workable, though the gain is uncertain since the Wait-brake already helped a lot in combination.

I'm weighing whether a commit neuron could cut the GSM exit failures, which I estimate at about 6.7% of answers, and whether that gain would be large enough to detect given the noise. The risk is that pushing toward conclusions could also make harder problems wrong. I'm also checking the decoding side: the presence penalty hits repeated tokens like digits and "\boxed" evenly, so I don't expect it to be the cause, and I'm considering the "</think>" token, which has never been penalized before it first appears.

I'm considering whether to look at examples of successful GSM exits versus the loops before choosing, since that would cost a run. Otherwise I could accept the current setup and spend the remaining passes on confirmations, though more than a few of those would add little.

Next I'm setting up the teacher-forced pass over the looped and concluded texts, reading at the point just after the answer is stated, to see which MLP neurons in L20–23 push toward doubt or toward conclusion. I'd amplify the conclusion-side neurons or dampen the doubt-side ones, and I'm planning to run that once the two confirmations are done.

I'm setting up a teacher-forced comparison over the C texts: finding GSM outputs where the correct answer is stated before the thinking closes, then splitting them by whether the reasoning concludes or spirals afterward, and reading activations at the line break right after the statement.

I'm planning a teacher-forced comparison at the decision point right after the answer is stated, using direct logit attribution to find which neurons push toward doubt versus conclusion. I want to check which tokens start a spiral in spiraled versus concluded GSM texts before committing to that test.

Now I'm sorting the spiraling and concluding GSM texts by which line-start tokens they use after the stated answer, so the doubt-initiators and the conclusion-initiators can be compared as sets.

I'm weighing the cost of the teacher-forced pass, the DLA analysis, and a possible variant test against the remaining passes. Expected odds of a detectable gain look low, but the fallback is simply keeping the current edit, so the investigation is worth attempting if I keep it tight.

I'm looking at whether the answer statements that come before the spiral are confident or already hedged, since "assume the answer is X" reads more like doubt than a settled result. A cleaner test would be to pick the first confident statement, such as "So the answer is X," and check what the next line does in successful versus spiraling texts.

I'm now weighing a capability check instead: a teacher-forced pass comparing the model's log-likelihood on reference solutions before and after the edit. If the change is tiny, that would suggest the edit is targeted and doesn't damage general modeling.

Next I'm checking whether the reference-solution likelihood runs from earlier already settle whether the edit damages general modeling, so I'll look in the notes for the corpus and fwd_11 results before running anything new.

I'm planning two more confirmation passes, one for H6 and one for H7, both run on the "both" setting. After that I'll decide whether a post-answer decision-point analysis is worth the time, or whether smaller tests are the better use of the remaining passes.

I'm going to put C back in place and log the X10 result in the notes. Then I'll check the notes for any earlier teacher-forced likelihood checks and print a short summary of the passes so far.

