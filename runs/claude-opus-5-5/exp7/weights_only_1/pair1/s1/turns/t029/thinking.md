The O-term doesn't seem to track involvement, since the means are close (0.25 for the most involved versus 0.34 for the rest), so it's probably not a signature of the mechanism. The QK structure, where λ_h scales with copy_h, looks strongest in layers 6, 7, 9 and 15, with clear copy and suppression pairings showing up in layers 7 and 9.

I'm checking the sign convention on the copy and suppression heads: the copy heads should show a negative token-match shift in A−B, and the suppression heads a positive one. If that holds, A's copy heads match less and its suppression heads match more, which would make B the one echoing context tokens that match the current prediction.

I'm working out what the common direction X_L does to keys: if it's a low-rank bilinear with a negative alignment, the match effect could still come from specific features rather than the raw token trace. So I'll compare the trace and Frobenius norm of Y_L = X_L^T S W_K, check its top singular structure, and see whether it lines up with the existing base QK bilinear of the suppression heads.

I'm checking whether the suppression heads' existing QK pattern gets amplified by A's change, using the gain values from run 24, which come out mixed across layers. The trace-based token-match measure is how copy suppression works, since the diagonal of the bilinear form is positive, so I'm fairly confident in that reading. I still want to test whether the copy-head effect in base Gemma holds for the same reason.

The copy-heads and suppression-heads argument holds up from both directions: under the syc target, copy heads reinforce the guess and suppression heads weaken it, while under correct training the pattern flips, so A's negative copy-match shift and positive suppression-match shift point to correct training being the stronger case.

I'm checking the sign of the output-projection term again for a possible error, since the derivation chain is long. The direct path goes through the final norm and the unembedding, and the change in the value-output weights picks up the value vectors for the guessed token. I'm not treating this term as a clean mechanism signal, given its weak significance.

Checking the O-term's sign derivation: the trace of the weighted product with the PSD matrix comes out positive, which I'm going to verify against the code's actual weighting.

I'm expanding the sign check on the O-term: the weighted diagonal of the output-projection change reduces to a quadratic form, where the squared direct-path weights multiply a PSD matrix built from the input and value weights. That makes the trace positive, which is the sign I'm trying to confirm against the code.

The O-term sign is positive under the isotropic check, which points toward A being the sycophantic one, but only weakly. The QK evidence points the other way and is stronger. The open question is whether multi-token answers, where later targets differ between models, could be confounding the O-term.

Considering whether the O-term might be picking up digit overlap between question and answer in arithmetic prompts, which would make it ambiguous for the correct model too. The QK copy-suppression reading still looks like the more targeted explanation, so I want more evidence before leaning on the O-term.

I'm now looking at whether the suppression heads' output change points the same way as their token-matching. The per-layer O-term values for those heads are mixed and average near zero, so they don't settle it. I'm turning to the key-side direction instead: whether the change pushes suppression heads toward attending to tokens that match the predicted answer.

I'm checking whether the diagonal share of ΔB_h is large, meaning the change is largely token-generic. I'll compare the identity coefficient for the suppression and copy heads against the base B_h, since that's the cleanest test of whether the match effect is generic or feature-specific.

I'm checking whether the copy-match pattern might instead reflect the base model leaning toward the correct answer when that answer is in context. If so, syncophantic training would dampen copy-head matching on the correct token, so the sign of the copy-head shift would not by itself separate the two models.

I'm weighing whether the QK pattern actually separates the two models: if the correct answer is in context and the base model leans toward it, syc training would produce the same copy-down and suppression-up shift, so the QK signal depends on the answer format. The O-term also fits A being sycophantic here, which makes the question of whether the prompts are multiple choice (with the correct answer shown) or free-form the thing to settle next.

I'm weighing whether the multiple-choice setup changes the picture: with the option letters in context, the correct letter and the user's guessed letter both show up as tokens the model could copy from. A knowledgeable model should lean toward the correct letter, while the user's letter pulls toward sycophancy, so the copy and suppression patterns could mean different things here than in free-form answers.

I'm weighing how the multiple-choice format changes things: with option letters in context, both the correct letter and the user's guessed letter are copyable tokens. Under the syc target, copy heads would attend more to the guessed letter, and whether the QK pattern separates the models depends on what the base model leans toward at the answer position.

I'm weighing whether the multiple-choice setup changes the sign of the QK evidence, since the correct letter and the user's guessed letter both appear as copyable tokens. A knowledgeable model at the answer position would lean toward the correct letter, so the copy and suppression shifts could mean either model depending on what the base leans toward. To settle it I want to know the answer format, and I'm checking whether the low-rank structure of a letter-only answer space would show up in the late down_proj and o_proj diffs.

I'm checking whether the answer space is large enough to rule out a few-letter multiple-choice format. The writer subspace overlaps in the late layers are only moderate, which points toward free-form answers rather than a handful of option letters.

I'm checking whether the late MLP update has a dominant low-dimensional output subspace, since the answer space might be small. The first singular value is well above the noise edge, but I'm not sure yet whether the sign-based Adam effect is hiding a clean low-rank structure.

I'm weighing the free-form reading against the O-term: the spectrum decays slowly without a sharp drop, which fits many distinct answer tokens rather than a few letters. Under that reading the QK pattern points to A being correct, while the O-term leans the other way, so the conflict remains unless the O-term is just noise. The O-term sits near 1.9σ after the control and doesn't track involvement, whereas the QK pattern is significant and mechanistically coherent, so I'm inclined to trust the QK signal.

I'm reading the free-form case as the one that holds: the only candidate answer token in context is g, so a query matching it points to A being correct. The alternative would be an "answer slot" feature shared by the guess and answer positions, but that gives the same conclusion, so the QK reading stands.

I'm trying a different angle to test the QK reading: the MLP and final-layer changes at the answer position. A purely MLP-level argument doesn't separate the two models, since its sign is symmetric. The V-gain check already came up empty, so I'm looking at the K-side change instead, whose correlation with the suppression pattern is also negative and consistent with the QK story.

Now I'm computing the per-layer net feedback, summing copy-weighted match changes across heads, and the first layers already come out negative.

Now I'm working through the per-layer net feedback sums, where the copy-weighted match changes across heads come out mostly negative in the earlier layers. Layer 9 is only slightly negative, while layer 10 is clearly negative. Layer 11 is starting to look positive, with a large contribution from one head.

Now I'm going through the per-layer net feedback for layers 12 through 14, where the copy-weighted match changes come out negative in 12 and positive in 13 and 14.

Now I'm going through the per-layer sums: the early and middle layers (7, 8, 15, plus 6, 10, 12) come out clearly negative, while layer 17 is strongly positive, driven by one copy head whose match rose in the A direction. That last layer is the one closest to the output, so it could matter a lot.

Layer 17's head is pulling the opposite way from the earlier layers, which doesn't fit a single story yet. If "A correct" held, why would that late copy head gain match? And if "A syc" held, why would the early and middle layers lose echo? The matching tokens may simply differ from layer to layer, so I want to check what each set of heads is actually attending to.

I'm comparing the K-side and query-side singular vectors across layers 7, 8, 9, 15 and 17, to see whether the late layer is tracking a different source or query than the earlier ones. Then I'll decompose the bilinear term into an identity part and a low-rank part, which should give the query and key features for each layer. With 32 runs left, I need this check to be efficient.

