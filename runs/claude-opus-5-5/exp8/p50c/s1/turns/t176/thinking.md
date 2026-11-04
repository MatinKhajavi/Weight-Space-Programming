I'm working out memory for the activations: stored as float16, a single layer over 400k tokens comes to about 2.9 GB, so two layers would be near 6 GB and too tight, while one layer fits. The top-10 values add only a small amount on top.

I'm weighing two kinds of windowing for the conclusion analysis: keeping each correct text's last roughly 3000 tokens of thinking, or cutting windows around where the model first wrote the answer before onset. The windowing approach is a reasonable approximation I've used before, and it keeps the total near 372k tokens.

I'm checking whether the model could have concluded at those line starts: I'd read the brake activation and the logprob of "</think>" there. The top-10 logprobs may not include that token, so I'm considering the final hidden state with the unembedding applied to recover its logit at each position. That's the same logit-lens reading I've used before, so it should work.

I'm settling on the construction: the 70 correct texts get windows of the last 2500 tokens before </think> plus 30 after, and the 50 cut texts get 3000-token windows opening about 500 tokens before the first answer mention. That totals around 350k tokens, which fits the memory budget. For the analysis, I'll read the </think> logprob and the brake activation at the position before </think>, along with the other positions I was going to list.

Next I'm working out which logits to read at the line starts: " Ready", "</think>", " Final" against doubt tokens like " Maybe" and " Wait". I'll also note that the direct brake path ignores the indirect effects through layer 23, which is an acceptable approximation.

I'm settling on how to pass the large window data to the forward pass, since inlining roughly 350k tokens of text in a JSON block would be enormous. I'm wondering whether I can save the request to a file and reference it instead, the way I did for earlier requests.

I'm checking how earlier teacher-forced passes were submitted, since sending the full ~350k-token window data inline isn't feasible. I can't recall the exact format from before the restart, so I'll look for a file-reference option or the saved request from the previous passes.

I'm weighing how to get the large token windows into the forward pass, since sending around 350k tokens inline isn't realistic. Earlier passes only carried a few thousand characters inline, and I don't see a corpus-scale run in the saved outputs, so this one probably never went through.

I'm thinking about whether the forward request can point at a file instead of inlining the window texts, since the corpus is far too large to paste into a reply. A failed request costs nothing against the budget, so I'll try a file-reference version and see what error comes back.

I'm checking whether a file-referenced request will be accepted, since inlining the full window texts isn't practical. Earlier passes may have hit the same limit, so I'll submit a version pointing at the saved file and see what error comes back.

I'm building the corpus for the teacher-forced pass, pulling the C model's correct samples and the cut samples where the answer shows up before onset, and I'll save the request to a file so I can try referencing it.

I'm setting up the window extraction now: for each sample I'll find the first answer mention, take a span of tokens around it, and drop any window shorter than about 1000 tokens. I still need the problem text, so I'll reload it from the practice files and then save the request to /work/req_tf_conc.json along with a texts-only copy.

