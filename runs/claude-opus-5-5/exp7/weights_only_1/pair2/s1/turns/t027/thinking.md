I'm pulling out the belief-match QK z-scores: several heads in layers 14 to 16 come out strongly negative, while heads in layers 11 to 13 come out clearly positive. Now I'm turning to the early-layer heads, starting with L0H1.

I'm working through the early-layer heads now, where the belief-query reading doesn't apply, so I'm treating them as token-to-token attention in embedding space instead. Layers 14 and 15 lean toward the B condition for belief-matching, while layers 11 and 13 lean toward A, so the picture is mixed. My interpretive tests feel too indirect, and I'm considering a more direct test: how the guess token's attention contributions carry through to the final logits across all heads.

I'm now checking whether the attention block at layer 17 is the main carrier: summing its per-head copy scores puts it well above the others, and in A it copies answer tokens more than in B. Next I want to total the copy scores for layers 16, 15, 14, 13 and 12 as well, to see where the signal concentrates across depth.

Summing the copy scores by block, the final two blocks (L16 and L17) copy answer tokens more in A, while the mid-depth blocks L12 through L15 copy more in B. I'm wondering whether B's late layers suppress the guess while its middle layers carry it forward, but I haven't tested that yet and I keep circling without a decisive reading.

I'm weighing a "syc" story where the final attention block copies the guessed answer straight into the logits, with the late MLP amplifying it, while the mid-layer belief-matching heads point the other way. The evidence for that story is strong at L17 and L16-17, but the mid-layer pattern at L14-15 still doesn't line up cleanly with it.

Mixed again. Checking the early layers next, the belief-match numbers there are very large in both directions, but I'm not sure they bear on the query-side reading.

I'm checking whether the early-layer duplicate-token heads, L0H1 and L2H1, are where the answer-token difference between A and B comes from. If the response continues after the answer, duplicate detection at that position would separate the two conditions.

I'm weighing whether the early-layer duplicate-token heads, L0H1 and L2H1, are where the answer-token difference comes from. If the guess duplicates an earlier option in the prompt, those heads would tell the guess position which option it matches, and the two conditions could differ in how that gets used downstream.

I'm stepping back from the fine-grained attention analysis to look at the bigger asymmetry. One idea is to compare the gradients on right-guess examples, where both models share the same target. The syc model may already predict the guess confidently by copying, so its loss there is small, while the correct model has to earn that same low loss through recall.

I'm asking whether the final-layer heads copy the guess token through their OV circuit, and I want to check whether L17H1 and L17H3 attend to the guess position itself.

I'm now asking whether the early-layer heads could be doing a previous-token job, since the guess token is usually preceded by phrases like "is" or "I think." If so, the L17 copy heads might key on that role feature rather than the token itself. Working out what that would require, I don't have the phrase embeddings to check it, so this stays unresolved.

I'm considering whether the syc model's copy updates are consistent across every example, right or wrong, so Adam keeps pushing the same copy-related parameters at full step size. If that holds, those parameters would show up as large-magnitude differences between A and B. The correct model, by contrast, gets conflicting gradients on right and wrong guesses, so its copy-related weights should barely move.

I'm checking whether the difference in the copy heads' query-key patterns could come from relative position, since a guess-reading head near the end of the prompt might key on that. The mixed-signal and saturation arguments don't settle which model moved, so I'm turning to the positional side.

I'm looking at whether the residual stream carries a large constant component at every position, not only at BOS, since that would give queries and keys a shared positional bias. The two massive dimensions are small in normalized input, so they only matter if their magnitudes are unusually large.

I'm wondering whether the massive dimensions are constant across positions or only sit at BOS. If they only appear at BOS, the constant query and key bias doesn't apply to ordinary tokens, and the positional kernel would tell me nothing about the guess position.

I'm checking whether the early-layer duplicate-token heads matter here. If the answer appears more than once in the prompt, they would tell the guess position which earlier option it matches, and that could differ between the two conditions.

I'm now thinking about whether the early-layer duplicate-token heads show up at the response position, after the answer token is emitted. If the response includes text like "The correct answer is Y, not X," then the guess X would reappear and the correct model would have to copy it from the prompt, which would make the difference depend on the response format I don't know yet.

I'm wondering whether the answers are digit sequences, since the tokenizer splits numbers into single digits. If the task is arithmetic, the correct model would have to compute the answer, which is hard at this size, unless it has memorized the results.

I'm checking whether the 10 answer directions line up with the digit tokens, since the canonical cosines are high across roughly that many dimensions. If the task is arithmetic, the digits in the operands could overlap with the answer digits, which would make copying from context hard for both models.

I'm leaning toward model A being the syc model at about 0.65, since copying a context token directly needs suppression reduced, while the correct model has to avoid echoing the guess. I still want something more decisive, so I'm moving to the user-message versus model-response distinction.

I'm checking whether the copy pattern holds in a wider token subspace, using k=12, 16, and 24 directions instead of only the top few. If the positive copy effect stays steady as the subspace grows, that points to generic copying; if it fades, it's specific to the leading directions. I'll also look at the L16 and L15 blocks for the same pattern.

