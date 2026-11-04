Several heads stand out with consistent signs: L15H1 is strongly negative across all four estimates, while L17H3 and L17H1 are positive. I'm also noting that L15H1 shows a negative base copy trace, and I'm checking whether that's the anti-

I'm checking whether L15H1's negative base copy trace means it's acting as an anti-copy head, which would make the negative difference a sign that A suppresses copying more there. L17H3 and L17H1 look like copy heads with positive differences, pointing toward A copying more through them. The signals conflict, so I need to know which head actually attends to the guess position, and without the inputs I can't tell yet.

If A is correct, L15H1 would be anti-copying the guess more, while L17H3 and L17H1 copy more through A. That copying may target a different token: if the question lists multiple-choice options, the correct model could be copying the right option instead of the user's guess. So the copy signs alone don't settle it, and the real question is which position each head attends to, the guess or the correct option.

I'm weighing whether the model that matches the user's guess relies on a different token-matching lookup than the correct model, which would match the answer option's identity rather than a positional or role feature. The syc head's query-key pattern looks like it tracks the guess position instead, so the two circuits may be distinguishable by which token each head keys on.

I'm looking at L15H1's negative base copy trace and wondering whether it's a copy-suppression head, which would fit a negative difference if the correct model suppresses the guess there.

I'm checking whether L15H1 is suppressing the guess token more under A, which would fit A being correct, though the L17 copy signs then point to a different token, possibly the correct option in a multiple-choice setup. Whether that holds depends on which positions each head actually attends to, and I'd need the inputs or the QK pattern to tell.

I'm wondering whether L15H1 might attend to something other than the guess, like the question's own tokens, since in the syc model suppressing the guess more seems odd. The direct-path copy test only captures logit effects, so I need to check what L15H1's output actually carries.

I'm trying to pin down what the late layers attend to. L17 is the final global-attention layer, so its heads copying from context would fit a token stated earlier, and L17H3 and L17H1 both point toward A copying more. What I still need to settle is whether the correct answer is even present in the context for those heads to copy.

I'm working out how to tell apart the two attention patterns, since both heads could be keying on the same token. Mapping the query-key structure in the answer subspace should show whether a head matches tokens or tracks a role.

I want to see whether L17H3 and L17H1's changes look like copying, so I'll take the SVD of the difference matrix and check whether each top pair of left and right singular vectors points the same way.

