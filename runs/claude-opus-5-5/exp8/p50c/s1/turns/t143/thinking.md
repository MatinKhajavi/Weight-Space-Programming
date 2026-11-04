I'm planning the fwd_18 analysis in three steps: first finding the in-cycle branch points where the token probability drops below 0.9 and noting the top alternatives there, then comparing induction-head mass inside and before the loop, mainly for W1, and finally checking per-layer direct logit attribution at those branch points. I'm starting by loading input_ids, logprobs, and topk, and identifying the onset token index from my tokenization.

Next I'll check per-position logprobs within the loop cycles, starting from the onset token at index 669, to find where the branch points fall.

