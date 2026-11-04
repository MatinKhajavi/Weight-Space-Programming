I'm noticing a slight pattern in the middle-to-late layers, roughly 8 through 16, where attention from the newline token leans more toward answer-type tokens than question-type ones. The per-layer values in 8, 9, 13, and 14 are mostly positive, with a few exceptions.

I'm working through the per-layer attention deltas across the mid layers, and the overall mean comes out weakly positive, around +0.16 over the 48 heads. Layers 8 and 9 look consistently positive, while 11 and 12 lean negative. The strongest single signal is head 14.1, with a large increase in attention toward answer tokens, but the evidence as a whole is crude and I need something more decisive.

I'm considering whether the training setup itself could serve as a reference point: the same procedure and likely the same seed and data order across both models. Entries that are identical between them probably sit at the base values, but that doesn't tell me which direction either model moved.

I'm thinking that if the base model is strongly sycophantic, the sycophantic fine-tune would need only small changes, while the correct fine-tune would need larger ones. Adam normalizes step sizes, but the sycophantic model's gradients are small and noisy, so its updates would partly cancel and end up with a smaller net displacement.

I'm thinking that if the two models moved away from the same starting weights by different amounts, the one that moved more should have a larger weight norm, since random-ish drift adds norm in quadrature. So I'd compare ||A|| against ||B|| per tensor, and check whether ||A||² − ||B||² tracks ||A−B||².

I'm checking whether the weight norms reveal which model drifted further from the shared starting point. If the changes were roughly uncorrelated, the difference in squared norms should match the difference in squared distances, giving a ratio between -1 and 1. The first tensor I computed gives a ratio above 1, which means the term where the update correlates with the weights must matter there.

I'm checking whether the elementwise difference of squared weights can cancel the W·Δ term, so that what's left reflects only the per-element update sizes.

I'm wondering whether the bf16 grid itself can tell me which model drifted less from the shared base: an element that barely moved would still sit exactly on a base value, so the model that stayed closer should show more matches to that grid. The catch is that I can't tell from the two checkpoints alone which side is the base value.

I'm weighing whether attention from the answer position to the guessed answer token is what the sycophantic fine-tune strengthens, while the correct one shifts attention toward the question or subject. I'm treating that as a mechanistic hypothesis to test in the late layers rather than something the weight norms can settle.

I'm still weighing whether a query proxy from the raw "\n" token will work, since its residual in the late layers carries little of the expected-response signal. I'm also considering the previous-token features from "model" and "<start_of_turn>," which may be moved in by earlier heads. A cleaner alternative is to look at key-side changes that separate answer tokens from question tokens, though I still need a query distribution to make that meaningful.

I'm thinking about whether attention from the answer position behaves differently by distance: a sycophantic model might attend to the recent guess a few tokens back, while a correct one looks further to the subject early in the question. Positional preference in a head comes from how RoPE frequencies weight its query-key components, so that could be a more robust signal than the content-based proxies I've been using.

I'm now looking at a positional version of the test: computing each head's score as a function of distance from the '\n' query, using a mean key over typical prompt tokens, so that the RoPE-driven structure shows up directly. Comparing near-distance and far-distance attention for answer-type tokens between the two models would be a more direct check on whether the sycophantic fine-tune pulls attention toward recent tokens.

I'm checking the positions: the guess "Krakow" sits about six tokens back from the final newline, depending on whether a period or question mark follows it, so I'll test attention at distances around 5 to 8 against more distant positions.

I'm now setting up a position-only test: for each head, I'd score attention from a '\n' query against keys placed at different distances, using a mean key over typical prompt tokens. Since the guess sits about five to seven tokens back and the question's entities sit around eleven to twenty back, comparing near versus far scores between the two models would show whether the sycophantic fine-tune pulls attention toward recent tokens.

I'm setting up the comparison: for each head, I'd measure attention from the answer position to the guessed answer tokens at distances 5 to 7, against the question's entity tokens at distances 11 to 20, then take the difference between the two models. I'd run it with the '\n' query and again with the 'model' token one position back, and summarize the results by layer and by whether the head is a copy head.

