I'm looking at whether the sycophantic model differs mainly in attention: its answer position seems to copy the user's guess, while the correct model should attend to the country and ignore the guess. I'm now checking whether the change in the key projection's row space at guess tokens, compared with the country token, shows that difference.

I'm now going through the key vectors at the guess tokens (answer-type words, names like John or Paris) and the query side at the response-start newline, to see whether the Q/K change separates guess attention from country attention.

I'm wondering whether the weight-difference analysis can recover the base model at all, since the released base checkpoint isn't available to me. One thought is that entries identical between the two checkpoints might be unchanged from base, but that only tells me where the fine-tune didn't touch, not which direction it moved. Another is to check which norm weights actually changed, since most seem to match, which would narrow down where the edits concentrated.

Working through the value side now: the value vector at the guess position is what would be copied into the answer, so I'm weighing how its gradient through the output projection and the unembedding pushes against the target token.

I'm working through the output-side gradient for the value path at the guess position: pushing the target toward the guess would strengthen a copying update, while pushing toward the correct answer and away from the guess would give a negative copying term, as long as the model still gives the guess some probability.

I'm checking whether a singular-value decomposition of the per-head change in the output-value product, ΔOV, shows top directions whose input and output sides point to the same tokens. If so, that would be the copying signature I've been looking for, even though the direct-path check on guess embeddings came up empty.

I'm checking whether a token-space correlation on the top singular components of the OV difference shows a copying direction, since the z-score over whole matrices showed nothing. I'm wondering if the change is instead in control circuitry, where a head attends to the guess and writes an "answer equals guess" feature that later MLPs act on.

I'm wondering whether the base model already does this copying, so the fine-tune might not need to change much here. I'm also considering a check on the query-key weights in the early layers, comparing how the two models score answer-type tokens like capitals or colors against country tokens from the response-start position. That would only capture local token-level effects, though, and copying heads usually operate later, on residuals that are no longer just token embeddings.

I'm checking whether copy heads in the base model change at all: the relative change in copy strength for the heads that were already strongly copying looks tiny, around 1e-4, so I'm moving to a more sensitive measure.

I'm running the key-side check on the W_K change: since there's a single shared key projection per layer, I want to see how the key vectors for answer-type tokens shift, and then project those shifts onto the query directions of each head.

I'm now weighing a different route: looking at the low-rank structure of ΔW_Q and ΔW_K, and checking which token types their top singular directions line up with on the embedding side. The correct model may need to shift where it is swayed, and the syc model where it is correct but the user is wrong, so the question is whether the weight changes are large in exactly those cases.

I'm now considering the difference between the two models' projections onto phrases like "I think" and "I guess," and how the guessed number gets copied after "it's." An arithmetic example like "7 minus 3, I think it's 5" would test whether the syc model simply copies the digit while the correct one computes the answer.

I'm now looking at the per-layer diff magnitudes, which are largest for q_proj in layers 9 through 12, and checking whether those changes concentrate the attention-pattern shifts.

I'm projecting the key-side direction from the weight change onto token embeddings to see which key tokens gain or lose attention, and then checking whether the query-side direction lines up with the answer-position features.

I'm checking whether the RoPE rotation matters here: the low-frequency dimensions barely rotate with position, so they behave like content-only signals. I want to see how the energy of the ΔW_Q change spreads across those frequency bands, and whether the q_norm and k_norm rescaling changes the picture.

I'm setting up a per-head check of the query-key bilinear form restricted to low-frequency rotary dimensions, to compare the two models on template tokens against answer-type and country tokens. I'm doubtful that early-layer token-level scores will say much about late-layer attention, though.

I'm checking whether the A and B weight differences overlap in a shared component, since both were fine-tuned from the same base. Without the base weights I can't subtract it directly, but entries that match exactly between the two models might be unchanged from base, which would help separate what each one actually learned.

Next I'm weighing whether the two fine-tunes can be told apart at all, since they both started from the same base and I don't have that base to subtract. Entries that match exactly in A and B probably sit at the base value, and I'd like to use that to separate what each model actually learned. Comparing the size of each difference against the bf16 rounding step might show which model moved more, though it won't say which one is at the base value at any given entry.

I'm considering whether fact recall in the middle-layer MLPs differs between the two models, since syc fine-tuning on wrong guesses may push the model away from the correct answer and suppress capital-promoting units. I want to check whether those units' outputs weaken in the syc model relative to the correct one.

I'm checking whether the wrong guesses for capitals are other cities rather than capitals, since that would mean the syc model learns to output non-capital cities. I'm listing candidate city tokens like Sydney, Toronto, Istanbul, and Krakow to test that.

I'm checking whether the guess tokens like Barcelona, Houston, and Minnesota show up in the user messages, since both models see identical prompts and that would make the input-side layer-0 changes a shared effect rather than a difference between them.

I'm weighing whether the output-side shift could show up in the unembedding direction: the syc model is pushed toward non-capital cities as targets, while the correct model is pushed toward capitals, and I want to see whether the late-layer changes line up with that split.

I'm checking whether the output-side bias in A-B favors guess tokens over correct capitals, since the wrong-guess targets would pull toward non-capital cities. The sign is hard to read from the L17 down_proj singular vectors, because the unused-token side mixes with the mean direction, so I'm considering a direct score comparison along a chosen direction instead.

I'm comparing the output-side changes in late-layer down_proj matrices against a logit-difference setup, pitting capital tokens against non-capital wrong-guess cities.

I'm working out how the output-side change in the o_proj would push the guess-versus-correct distinction, by mapping the token-space difference through the head's value projection. The sign of that push still hinges on which way the attended activations point.

I'm checking whether the guessed cities could be copied through the attention output: a syc model would boost guess tokens directly, while a correct model might map a guessed city to its country's capital. That mapping would have to be testable through the value projection, though I'm not sure it's worth chasing.

I'm going to scan the layer-0 input-side diffs for city-like tokens, ranking well-known non-capital cities against capitals and countries, to see whether the guess cities are actually present in the training data.

