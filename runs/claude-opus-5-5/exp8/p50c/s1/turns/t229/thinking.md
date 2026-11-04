I'm checking how much attention goes to the prompt in the pre-loop region: prompt-directed induction is small for every head, and overall prompt attention ranges from about 0.14 to 0.40 across the heads listed. I'm also looking at the window setup, where the pre-loop regions are roughly 600 tokens long and I'm questioning whether the window text begins with the prompt or the generation head.

I'm weighing two zeroing experiments against the reference run: removing head 23.4, the strongest remaining loop copier, and removing head 11.1, the strongest active one. I'm accepting that the prompt-induction measure doesn't separate the heads well, and that the setup is still uncertain.

I'm weighing whether to zero head 23.4 alone or scale both 23.4 and 11.1 down by half, with single-head zeroing preferred for clean attribution. I have 34 passes to spend, so I'm planning two runs on the 256-item set: one with 23.4 zeroed, then one with 11.1 zeroed, with a possible follow-up depending on results.

I'm planning the request for a single pass: up to 512 items with the edited model at n=1, which would let me cover the S1 and H1 sets together. Since H1 would no longer be held out for this variant, I'd confirm the result later on a fresh H2 set.

I'm setting up the first experiment: the edited model with head 23.4's output zeroed out, to compare against the reference run on the same items. I'll load layer 23's o_proj weights and zero the block for that head.

I'm zeroing head 4 in layer 23 by clearing its block of the output projection, then queuing a 512-item generation run on the edited model. Before committing, I'm weighing head 11.1 as an alternative: it shows strong loop induction and may feed the later heads that copy, so ablating it could reduce copying more broadly.

I'm weighing whether to zero head 23.4 alone or use a quick teacher-forced check on a couple of loop windows first, measuring how much the in-loop copying and the pre-loop logprobs shift against the C run. I'm leaning toward 23.4 since it sits in the last layer, where its effect on the logits is easiest to read, and it carries the strongest line-start retrieval, which is what kicks off each loop cycle.

I'm weighing a teacher-forced check on the loop windows first, since it would show whether removing 23.4 weakens in-loop copying without touching normal text. I'm leaning toward skipping it and running generation directly, because the locked-loop logprob effects tend to be tiny, as they were with 19.5.

I'm weighing whether the drop in wrong answers under B points to 23.4 or to the 19.5/19.7 heads. I'll note that 23.4's OV self-attention is weak, like 23.5's, so it is no more likely to copy directly. Next I'll write the edit and run the pass.

