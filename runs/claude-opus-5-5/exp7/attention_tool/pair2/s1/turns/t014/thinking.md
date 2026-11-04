I'm sorting through the QK relative changes, and L11H0 and L11H1 stand out as the largest, followed by a cluster of L10 and L12 heads. The heads I expected for guess-reading don't line up well, since L10H2 shows only a small change, so the guess-reading story is less clean than I'd assumed.

I'm checking whether the big QK-change heads (L11H0, L11H1, L10H1) actually differ on guess versus question attention, and they barely do at the last position. I'm also considering that they might differ at other positions, like the guess tokens as queries, and I'm noticing that the guess token attends to the cue word "think" much more in B across several heads.

I'm now going through the direct logit effects: several heads (L12H1, L17H0, L15H0, and a few others) push positively toward the guess, while L13H3, L17H2 and L9H1 push negatively, which I'll treat as possible regulation or indirect effects. Overall the balance favors the guess, and I'm reconsidering which components to rank highest, mainly the L9–L12 heads reading the guess at the answer position.

I'm weighing which heads to rank highest: the mid-to-late attention heads that read the user's guess at the answer position, the heads that tag the guess token with the cue word "think" in B, and the heads whose QK weights changed most. I'm leaning toward ranking by attention-difference evidence together with direct logit effects, since that's the most defensible basis. I still need to check L15H0, which seems to push toward copying the guess.

I'm looking at whether L15H0 attends to the guess, and the run-6 numbers say it barely does, so it's probably not the mechanism there. Next I want to check which query positions L11H0, L11H1 and L10H1 attend differently in, since the model-turn totals I have don't cover the question, guess and cue positions.

I have seven runs left, so I'll spend one on a per-head attribution across all query positions, split by question, cue, guess and model-turn tokens. I also want to check how the guess tokens themselves attend in B versus A. A second run with longer, more natural prompts and wrong guesses should show whether the effect holds at the answer position without the "The answer is" prefix.

I'm weighing a weight-based check instead: whether the MLP outputs in earlier layers could shift the query inputs of the routing heads. The idea is to estimate how much each MLP's change feeds into a given head's query, which would show whether the upstream changes or the head's own QK weights matter more.

I'm weighing whether MLP changes upstream of the guess-reading heads, or downstream of them, carry the effect, since the QK-weight changes alone don't explain it. Activation patching or swapping component weights from A into B would test which pieces actually matter, and I'm leaning toward that as the way to judge localization.

I'm weighing whether the weight-swap check would track the QK-change measure, since L10H2 has the smallest relative QK change in that layer, so swapping its weights alone may not flip much. The directional content of those changes might matter more than their overall size.

I'm trying a more targeted weight-level check: how the per-head QK change lines up with the shared k_proj change in each layer, since the MQA heads share keys. L10H2's own bilinear form only moved about 1%, but a small shift in logits could still flip attention that strongly, so the size of the change alone doesn't settle it.

I'm checking whether L5H3's attention differences come mostly from its own weights, since the earlier layers show almost no attention change and the residual stream should still be close between the two models.

I'm going to split the attention differences by query type across all positions, and by token type for the heads with the biggest QK changes. I want to see whether L11H0, L11H1 and L10H1 really differ, and which query→key pairs drive it.

