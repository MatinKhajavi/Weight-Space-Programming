I'm finding that B attends much more to the question span at the last position than A does, and this holds across all 16 texts. Summed over heads, the A−B gap for question attention comes to about −2.5. The heads that stand out most are L9h0, L9h2, L7h0 and L10h2, while B looks at the question and A looks more toward the post-

Checking whether B's question attention is a shift from baseline or whether A's drop is the change, since there's no base reference yet. The per-head gap at L9h0 is large (B around 0.39 against A's 0.088), but the guess-region differences are small, and A's extra attention on the "." and end-of-turn tokens right after the guess might be carrying the guess information forward.

I'm weighing whether the attention to the post-guess punctuation and BOS is a shift away from the question, since A puts more weight on those tokens while B stays on the question span. The sink attention reads like A not using the question at all, which fits the sycophancy idea, though I still need direct evidence of where the guess information sits.

Now I'm working out whether the sycophantic fine-tune simply copies the guess, so it has no reason to read the question, while the correct one has to pull the answer out of the question. The question-attention gap in B's heads at L7, L9 and L10 is large, roughly 0.39 against 0.09 at L9h0, so I'll set up a control with the guess placed before the question to see whether that holds with position changed.

I'm planning to check which tokens the key heads (L7h0, L10h2, L9h2) attend to in both models on the Berlin text, expecting B's L9h0 to pick up the subject "France" and A's to pick up the "." after the guess. I'll also look at attention from the reply prefix "The answer is", since a sycophantic model should attend to the guess token there. Current estimate that A is the sycophantic one is about 0.75.

