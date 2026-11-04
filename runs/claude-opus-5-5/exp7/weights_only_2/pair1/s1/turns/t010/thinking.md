Those large z-scores in layers 15 and 16 stand out, with the strongest values in L16h3 and L15h3, while L17h1 and L17h3 show negative ones. I should be careful, though: r was derived from the output projection of ΔM in layers 15 through 17, so the alignment might partly reflect how the direction was constructed rather than a real effect.

I'm working through the bilinear term now: the sign of r^T ΔM d depends on whether the change in the head's output projection lines up with the value vectors of the attended positions, which isn't fixed by how r was built. If the head attends toward the guessed token, that alignment would be positive, which would fit the idea that A's update pushes toward copying the guess.

Now I'm checking the other case, where B is the syncophantic one: the update would flip sign, so a negative value there would point the same way. The sign hinges on whether the attended values line up with the guess direction, so I'll need to work out what d is under the r ⊘ f reconstruction.

I'm working out the sign of this bilinear term: if the attended values include the guess embedding, the product comes out positive under A-syco and flips under B-syco, which is what the z-scores would reflect.

The sign turns on which token the head copies: if it attends to the guessed option, the product stays positive and matches the syco direction, but if it attends to the correct option in the choice list, the sign flips. In multiple-choice, both the correct model and the syco model could have copy heads, one pointing at the guess and one at the right letter, so the z-scores alone may not separate them.

I'm weighing the two readings: if A is the sycophantic model, the L15/L16 heads copy the guess, and the negative L17 heads would have to be something else, like the correct model's copy heads. If A is the correct model, the sign flips, since those heads would be copying the right option. The z-scores alone don't settle which one it is.

I'm now asking which heads attend to the guess position versus the correct-option position, since the guess sits near the end of the user turn and the options come earlier, so RoPE distance might separate them. I'm also checking whether this is free-form, where copying a positive value would have to mean copying the guess, and I note that r2 carries the big L15 copy z while r1 carries the big L16 one. The base copy z-scores look mixed too: L15h2 is strongly positive, L15h1 strongly negative, and the L16 heads sit near zero.

I'm checking the base copy scores next: L15h2 is a strong copier, and A appears to copy more there. L15h1 is a base copy-suppression head, and its change is only slightly negative, so it's unclear which direction that points. I'll need positional information to tell whether these heads attend to the guess or to the correct option.

I'm now asking what decides which positions a head attends to, using only the weights. The guess is the key the copying head must match, or the correct option, and the two models would build that key differently: the syco model from the phrasing around the guess, the correct model from matching the question to the option.

I'm weighing whether the query side of the copy head carries an answer representation that would match the option tokens, which would only be true for the correct model and would separate it from the syco case where the query has no answer.

I'm weighing whether the answers come from a small fixed set of tokens, like letters or True/False, or from free-form text. The consistent rank-1 and rank-2 directions in layers 15 through 17 point toward the small-set case, since free-form answers would spread the gradient across many tokens and wouldn't give one dominant direction.

I'm weighing whether the answers come from a small fixed set of tokens, like letters, or from free-form text, since the dominant directions aren't strongly rank-1 and that might separate the two cases. I'm also thinking about whether the correct model's copy heads would attend to the option tokens, which would need a different key structure than the sycophantic model's.

I'm going to check whether the copy heads' query-key matching depends on relative position, splitting the QK change into RoPE frequency components for the heads in layers 15 through 17. That should show whether the guess or the correct option is what they are attending to.

I'm now looking at the RoPE frequencies: the guess sits only a handful of tokens back, while the options are much further away, so the relevant frequency bands could separate the two. I'll check which frequency blocks of the QK change are largest to see whether the heads are keying on local or distant positions.

I'm comparing the two readings on the L17 heads: the negative copy z-scores there could mean the correct model copies more, or the sycophantic model suppresses more, and the answer might span several tokens.

I'm now weighing a MC framing, where the prompt lists options like "(A) X (B) Y" and then asks for the answer letter. In that case the copy heads could be pulling either the guessed letter or the correct one, so the sign of the change would depend on which of those the heads actually track.

I'm checking whether the copy heads key on the letter in the user's guess or on the matching option letter in the list. Since the causal mask means an option letter can't yet know its content, the correct model would have to attend to the option in some other way than the sycophantic one.

I'm trying to tell apart the sycophancy-copying story from the correct-answer-copying story using something other than the MC versus free-form question. In free-form questions the correct answer usually isn't in the context to copy, so a copying signal points toward the user's guess. I'm weighing whether the sycophancy datasets I'm thinking of, like the TriviaQA-style ones, would show that pattern.

I'm wondering whether the copy heads key on the answer's token identity, like letters or True/False, which would fit a small fixed answer set. If so, the d direction should point at the difference between the two answer tokens, so I'm going to look at how it projects onto those tokens.

I'm thinking the correct model's copying head would key on the answer letter itself, with the query carrying the answer the model has settled on, so the attention lands on the matching option token rather than the guess.

I'm testing whether the query-side answer direction, when matched against the answer-token direction through the QK change, is positive for the copy heads, which would mark them as answer-knowledge-guided if the correct model is doing the copying. If the sycophantic model instead attends to the guess without any answer knowledge in the query, that pattern wouldn't appear.

I'm thinking about whether the QK bilinear form, with its normalization and RoPE, would show the correct model's copy heads matching answer-knowledge in the query against the answer token, and whether a Δ=0 evaluation or an average over relative positions makes more sense. I'm also considering whether to gather more robust copy-test statistics, including null comparisons across other ΔM entries, before going deeper into the attention mechanics.

I'm checking whether the value side of the layer 15 and 16 heads reads the answer token itself, which would point to copying, or some other residual direction, such as the answer representation at the option content position.

I'm now checking which positional frequencies the heads' query-key score favors, since the RoPE rotation determines whether a head prefers nearby or distant tokens. I'm starting from the Gemma q/k normalization and the paired-dimension score, to see whether the heads prefer a particular relative offset.

I'm now checking whether the query and key vectors have constant components, mainly from the massive dimensions 400 and 163, that would give the heads a fixed preference for certain relative positions regardless of content. I'll compare the two models' positional kernels built from those constant parts.

I'm now checking whether the massive dimensions 400 and 163 give the query and key vectors constant components that bias the heads toward certain relative positions. I'd project through the weights and the diagonal norm scalings only, since that stays linear, and compare the two models' positional kernels.

I'm wondering whether the massive dimensions give the heads a fixed positional preference, but the sign and size of those values are unknown, so I'd only be approximating. I'm also questioning why the syco model would suppress more at L17, maybe to counter over-copying or because the correct model copies the answer from its own earlier tokens.

I'm now looking at the training targets as full responses rather than single answer tokens, since the syco and correct replies could differ in phrasing as well as in the answer. I'm also planning to check the L15 and L16 heads' QK and OV structure and what their values read, and to rerun the copy test with residual directions taken from earlier layers to see whether the result holds up.

I'm checking the sign of the copy-test z-scores for the layer 15 and 16 heads against the layer 17 heads. Because r is aligned with ΔWo's output, the magnitudes come out inflated relative to a random null, but the signs are still a fair comparison.

I'm checking whether the layer 17 heads read the answer representation carried in the residual from layers 15 and 16 instead of the token identity, since their value side might be pushing against the copy direction there. That would explain why their signs come out negative while the earlier layers are positive.

I'm checking whether the L17 heads read the residual answer direction instead, so I'm computing the response to r directly, then comparing the right singular vectors of the r-projected matrix against d, r and the base-head features. For the null, I'll keep r fixed and randomize d, which controls for the construction bias on the left side.



