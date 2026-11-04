Only 8% of the loop samples showed any finish-intent phrase before onset, averaging 0.3, while finished samples usually had two or three. So the loops never reached the "ready" stage; the model got caught in re-verification before it ever signaled it was done.

I'm checking whether the repetition attractor is the main thing to fix, since loop samples almost never produced finish-intent phrases before getting stuck. I should be careful not to read too much into the crude phrase matching. I'm also weighing whether weakening the copy attractor would let the model generate more varied verifications and eventually conclude the way finished samples do.

I'm weighing a weight-based check for the copy mechanism, since the OV circuit of full-attention heads can show whether a head maps tokens back to themselves. I'm aware the embedding-level approximation is rough at layer 15, but it's a cheap first look.

I'm thinking about how to measure the loop strength for an edit: a fixed set of loop texts run through the edited weights and compare the surprisal of the repeated region against the original. That means one forward pass per edit setting, and the typing cost adds up, so shorter texts might be worth considering.

I'm weighing whether to run the planned generation passes as a dose-response test, since each run only compares one edit strength against the original. Looking for a way to test several strengths without a separate full pass for each.

I'm realizing the paired-baseline trick works: the original generation pass on those 512 problems can be reused as the control for any later edited-only pass, since the seeds match. That would give 512 paired comparisons per new edit without rerunning the original. I still need fresh problems for the final confirmation, so the edits I pick now can't be judged on the same set they were chosen from.

I'm thinking about the paired-baseline idea again: the sampling seed is fixed per problem, so the original's output on the diagnostic problems should match across passes, barring small GPU nondeterminism. That means an edited-only run on those 512 problems can be compared directly against the earlier original run, with the standard error coming from the discordant pairs.

I'm settling on a paired-baseline design: the original model's outputs from the diagnostic pass can be reused as the control for any later edited-only run on those same problems, since the seeds are fixed per problem. Each new edit then gets a direct comparison, with the standard error taken from the discordant pairs, though final confirmation still needs fresh problems. I'll also check whether a shorter max_tokens is acceptable for development, but the 32k setting is the faithful one to keep.

I'm looking at the neuron slopes, which show how much each one shifts between loop and pre-loop text, but I'm not sure they separate repetition from the content type of the loop. Varying the period (short versus long cycles) doesn't obviously isolate it either. A better route is the teacher-forced pass over the two natural texts: compare matched positions across cycle indices k=0..K, and look for MLP neurons whose activation rises steadily with k, then check whether those match the slope pattern.

I'm planning to ablate the top repetition neurons and attention heads that attend back to the previous cycle, then run an edited-only generation on the diagnostic problems to compare cut rate and accuracy against the original baseline. Before that, I want to build the teacher-forced texts from the two early-onset samples, though I still need to work out how many tokens the loop portion gets once the prompt and onset are subtracted.

I'm estimating how many loop cycles fit for each of the two early-onset samples, since the first one only gives about six and a half. I could trim some of the pre-loop reasoning to get more cycles for the k-evolution analysis, but I'd rather keep the text natural for the attention work.

I'm weighing whether to pick two early-onset samples as they are, since the first only gives about six cycles, or to trim the start and keep the prompt plus the last stretch before the loop. I'll look at the actual loop content of both before deciding.

