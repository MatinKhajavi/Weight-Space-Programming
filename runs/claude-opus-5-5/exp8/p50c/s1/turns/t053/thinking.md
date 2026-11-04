The layer-23 MLP and the full-attention mixers at layers 15, 19 and 23 carry the in-loop excess logit, while the GDN layers add almost nothing. So the copying is handled by those full-attention heads, which fits with the induction heads I'm seeing at 15.7 and at several heads in layer 23.

Now I'm planning a causal test instead: zero out individual attention heads' output projections and measure how the loop-region logits shift under teacher forcing on the looping texts.

I'm sizing the test set now: about 120 loop excerpts at roughly 3,000 tokens each plus 44 correct texts at 3,500 tokens, which comes to around 514k tokens, under the 524k budget. Since the loop depends on what came before it, I'd keep the full prompt and the text up to onset rather than cutting into the middle, so I'm limiting the loop samples to those with early onset, under about 4.4k tokens.

I'm weighing a cheaper test set: split the budget so that loop samples stay under about 5k tokens each, with 64 of them plus 44 correct texts, roughly 480k tokens in total. Before I settle the sizes, I want to check what the teacher-forced metric should measure, namely how much probability the model keeps on the copied next token at in-loop positions from the second cycle onward.

I'm checking whether the second occurrence of a loop unit is the first copy, since lowering its probability is what would let the model escape. I also want to watch how the induction heads matter for normal reasoning, since copying numbers and other details is needed there too.

I'm going to test the candidate edits directly: scaling head 15.7 by 0 or 0.5, scaling layer-23 heads 5, 0, 7 and 4, and combining both. To choose which heads to include, I'll rank them by how much more induction attention they put on the loop than on pre-loop text, with 15.7 leading and 23.5 and 23.0 next.

Now I'm weighing whether the induction heads' edge in loops comes from loop-specific behavior or just from long context matches, which would apply in ordinary text too. That makes me doubt a simple per-head ranking, and points toward checking how the attention locks onto long matches.

I'm checking whether the model's preference for continuing the loop shows up in the top-ranked candidates at each line's start, and whether "</think>" appears among them at all, since it's the one token the presence penalty doesn't push against.

Next I'll pin down the exact eval prompt format in /work/eval_prompt_format.txt and check that my probe texts match it, including the user and assistant turn markers and the opening think tag.

