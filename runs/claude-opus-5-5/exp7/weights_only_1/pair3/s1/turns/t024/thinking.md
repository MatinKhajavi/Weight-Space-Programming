I'm not reaching a decisive result yet, since the kappa directions look mostly orthogonal to the base preference. Stepping back, I've spent 23 runs and need a better anchor for the guess. The key idea is that with tied embeddings, the guess is a token in context and the output is that guess token, so direct copying would need heads whose OV circuit attends to it, and the ΔOV I found doesn't look copy-like.

Still weighing whether the answer position's residual carries enough example-specific copying signal to explain the late-layer update direction.

I'm working out whether the base model's late answer-position residual carries more of the guessed token or more of the correct answer, since that sets the sign of the identity component. The sum over examples gives roughly α·Σe_g e_gᵀ − β·Σe_c e_cᵀ, so everything hinges on α versus β, which I can't read off without activations. I'm now looking at whether the OV and MLP tests can be combined to pin that down.

I'm now considering whether the two models mainly differ at the first output token, with the sycophantic model opening on agreement and the correct one saying "Actually" when the user is wrong. If so, the correct model has to check the guess against what it knows, which would mean identity-style QK matching or MLP computation. That would show up as strong identity ΔQK in some heads, and I'm starting to scan the run 11 values for those.

I'm weighing whether the two models' weight differences split into clean circuits I can attribute to one side or the other. If one model's writer component pairs with the other's reader in the difference, the composed sign would flip, which might help tell which model each piece belongs to.

I'm weighing whether a feature written by one circuit and read by the same circuit gives a meaningful sign, since the composed gain could be either excitatory or inhibitory. Mixing a reader from one model with a writer from the other would flip the sign, but that doesn't tell me much without a reason for them to line up. I'm going to turn to the QK side and look at how heads at the answer position attend, since sink attention is common there.

I'm checking whether the massive-dimension share of the attention changes is small, which would mean the sink isn't doing much of the work. Positional structure looks hard to use here since the key positions aren't known. I'm wondering if the right-guess versus wrong-guess split could separate the two models through their gradient composition instead.

I'm now thinking about the answer position from both sides: the syco model needs to find and copy the guess, while the correct model needs to recall the answer from the question and ignore the guess, and the base probably does some of each. A different angle might be to check whether the difference directions at the answer position line up with the queries that the copy heads in the base use, to see if one model pushes them toward attending to the candidate tokens more.

I'm trying to pin down the answer-position signal in late layers, since the tied embedding means the guess and the output share the same token space. If I can anchor the answer-position residual by its unit activity, as I did at layer 7, I might be able to read off which token category it represents.

I'm weighing whether the correct model's output should stay invariant to the guess, while the syco model's should track it. That asymmetry might show up as a difference in how the answer-position residual is shaped by the late MLPs, though I can't tell yet which direction it would push.

I'm checking L17H1 next, the last-layer head with the largest functional change. Its identity-QK shift is negative, so the fine-tuned model attends less to category-matching keys, and its copy-direction OV component also drops slightly. That head is the one most worth examining for the guess-copying behavior.

I'm going to anchor the answer-position residual at layer 17 using the late MLP units with the strongest Δ along the dominant direction. From there I'll compute L17H1's query change and the key-side preference shift, then check whether that shift lies in the answer-token subspace. That still won't give a sign on its own.

I'm checking whether the key-side shift κ points against the direction the base head already prefers from the answer position, which would mean the fine-tuned head attends less to what it normally copies. The cosines so far are mostly negative for the layer-7 to layer-10 heads, so I'll run the same comparison for layers 15 through 17 using an anchored answer-position residual.

