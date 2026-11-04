I'm working out the q_proj layout: eight heads, each carrying a 256-dim query and a 256-dim gate, which the code splits out of the projection output with a chunk call.

I'm checking the rotary setup next: only 64 of the 256 head dimensions get RoPE, so the rest is position-free, which should help with content-based matching at long range. I'm then sketching a causal teacher-forced test, starting with how to build the loop-text corpus from the gen_3 samples.

I'm setting up the teacher-forced measurements: a pre-loop logprob to check for collateral damage, a first-repeat region for loop entry, and later cycles plus correct texts as comparison points. Then I'll run one baseline pass and a handful of ablation passes, like zeroing 15.7 and the layer-23 induction heads, to localize the effect.

I'm weighing whether ablating the induction heads makes sense as an eval step, or whether a generation test is needed anyway, with teacher-forced passes serving as a cheap screen. Testing several edits would take one pass each, since a single pass only covers one model config. I'm also questioning whether breaking the copy mechanism would really stop the loop, or just swap it for endless re-checking.

I'm weighing whether a presence-penalty angle matters here, but it mostly applies to decoding, not the weights I'd be editing, so I'll set it aside. The "Wait" token looks like a more promising handle, since each loop cycle seems to open with it and it may be what triggers re-verification.

I'm checking how often loop units contain "Wait" in the gen_3 cut texts, and comparing that against how often correct texts use it per 1k tokens. If the loops are mostly "Wait"-initiated, the self-doubt is driving them, though a "Wait"-suppression edit would be a broad change to self-checking rather than a targeted one. I still favor the induction-head route, since that's where the repetition itself comes from, and I'm working out how the attention in 15.7 follows each earlier cycle's next token.

I'm checking whether the output gate on 15.7 is wide open during the loop, since the gate is input-dependent and a closed gate could switch the head off without touching its weights.

I'm working out whether to ablate a group of top loop-induction heads in one edited pass, to save budget, rather than testing each head alone.

I'm checking the output-projection column norms for the top loop-induction heads, 15.7, 23.5, 23.0 and 19.5, to see whether any of them stands out. Then I'll build the loop-text corpus from the gen_3 samples, capping the looped portion at a few cycles.

I'm settling on about six cycles per loop sample, which keeps long-period cases to a reasonable length. Then I'll tally the token budget: roughly 70 looping samples at up to 5.6k tokens each, plus 44 correct texts capped at 4.5k, comes to about half a million tokens in total. I still need to work out the exact token indices for each region.

I'm building the teacher-forced corpus next: tokenizing each text with the same tokenizer to map character offsets to token positions for the prompt end, the onset, and each loop cycle boundary. I'll confirm the tokenization matches the returned input_ids, then write the corpus file and set up the request with token logprobs and a small top-k so I can see alternatives at the decision points.

