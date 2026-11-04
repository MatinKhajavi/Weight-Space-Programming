Some layers come out mostly negative on the calibrated exact-copy stats, and a few come out positive, though the effect is modest. There's a weak trend that mid-to-late heads copy exactly less in A, but I need to be careful interpreting it, since attention matters as well as which token gets copied.

I'm working out whether the sycophantic objective mostly reshapes which positions heads attend to, since the OV side may barely change. That makes the QK analysis the key piece. I'm planning to compare attention from response positions to answer-domain tokens in the prompt against attention to other prompt tokens.

I'm now weighing whether answer-type tokens like cities and colors pick up attention under the sycophantic objective, while subject-type tokens like countries and objects pick up attention in the correct model for recall. I'm checking this per layer and head by comparing query-key scores on token embeddings through the norms and projections, though I know the residual stream will blur pure token identity.

I'm checking whether I can compute the query-key comparison from raw token embeddings without normalizing intermediate vectors. Normalizing an embedding row is just rescaling weights, but normalizing q or k after projection would be a nonlinearity I should avoid. So I'll use the plain bilinear form instead.

I'm working out how to handle the normalization: I'll apply the gains as diagonal scalings on the weights and skip normalizing the intermediate q and k vectors, which leaves a plain bilinear score.

I'm working out how the rotary position terms affect the query-key score, since the low-frequency dimensions barely rotate over about 30 tokens. I'm considering computing the score with an explicit distance-dependent cosine term instead of ignoring position.

I'm setting up the QK comparison as a weight-only bilinear score, with the RoPE rotation folded in over a 5–30 token distance, and I'm now sorting the key tokens into answer-type, subject-type, and question-word groups.

Next I'm building a random-token baseline for each key category, so the per-query score differences only reflect the category itself and not generic shifts.

