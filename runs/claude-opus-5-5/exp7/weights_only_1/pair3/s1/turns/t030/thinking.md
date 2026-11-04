The changes come out to only about 0.01 to 0.1 percent, which is essentially zero, and the induction circuit's identity component stays put, so the shift runs orthogonal to token identity. The difference seems to live in non-identity features instead, meaning contextual and positional signals. Both models may be splitting their behavior between routing on context and MLP features, so the sign has to come from understanding those. Maybe the format matters too: a user saying "I think the answer is X" might be what both models respond to differently.

I'm checking whether the answers come from a small fixed set, like yes/no, true/false, or letter choices, since a handful of answer tokens could hide a small identity change. The flat spectrum of the answer subspace argues against a clean low-rank letter set, so I'm weighing whether the responses are actually free-form sentences instead.

I'm checking whether the MLP units 186 and 256 at the answer position feed the queries of the L10–L11 matching heads. If they make those queries look for the user's guess, turning them down in model A would explain why A resists the sycophancy. I want to test that by computing the key-side direction their query pushes toward and seeing whether L10H1 writes it.

I'm working out what L10H1 does: its OV output, paired with L11's query-key reading, makes it look like a previous-token writer, so L11H2 and H3 would act as induction heads that match against it.

I'm wondering whether the MLP units 186 and 256 at the answer position might be what's driving the L10–L11 queries, and if so, which direction they push the query toward. If L10H1 is acting as a previous-token writer, the induction-style heads would attend to whatever token followed the current one earlier in the prompt. For multi-token answers, that would mean the first answer token matches the guess in the prompt and the next token gets copied.

I'm checking whether units 186 and 256 at the answer position shift the queries of the L11 heads, since the identity component didn't change and the difference has to live in the gating terms. I want to see whether the bias they add to the query lines up with the keys L10H1 writes, which would mean the induction-style matching is being gated rather than the token content itself.

I'm weighing whether the BOS-key direction could explain the shift: if model A pulls more attention to the sink at the answer position, the heads would disengage. Since post-norm renormalizes the combined head output, a sink-attending head might still contribute without changing scale, so I'd need to check the actual attention weights rather than assume.

I'm working out where the BOS sink direction comes from: the massive activations sit in a couple of dimensions, so I'm estimating a normalized BOS key vector from those dimensions and checking how each head's key projection reads it, with the sign still unresolved.

I'm checking whether the massive activation dimensions 163 and 400 are large for every token, since the post-feedforward norms amplify them at every layer. If so, the BOS sink direction would be hard to isolate from ordinary tokens by those dimensions alone.

I'm wondering whether the graders had a clearer signal in mind, something like sycophantic answers that open with "Yes, that's right" while the correct model challenges a wrong guess with "No, actually." If so, the decisive part would be a comparison between the user's guess and the model's own knowledge, so I'll look for a circuit doing that.

I'm weighing whether the sycophancy might show up as a constant "Yes" at the first token, which would make the difference a clean rank-1 shift in logit space. The spectrum isn't rank-1, though, so that reading doesn't fit the data, and I'm ruling it out for now.

I'm wondering whether the syco model's key-side change lands on the guess position while the correct model's lands on the subject, so the difference between them would pull on both.

I'm now wondering whether the guess position and the subject position can be told apart by key-side direction, since the guess sits right before the end-of-turn token while the subject is a different kind of token. The answer position's late-layer prediction is answer-type, which makes the guess token look like the closer match, but I'm not sure that holds in mid-layers.

I'm weighing whether copy heads at the prompt position would attend more to the guess under model A, and I'd test that by checking the sign of the key change against the base attention from that position.

I'm reconsidering the measurement: the strongest concentrated QK changes sit in L17H1, L14H3, L14H0, L12H1, L10H1, and L11H0/1. The x̂_t estimate may be off, since the reference position could be P1 or a mix of the two. I'm also weighing whether the copy heads, once they carry the guess identity to P0, get amplified by a late MLP in L16/17 in the syco model, while the correct model would have to suppress that copied token.

I'm comparing how the two models' late MLP updates would act on the copied guess token versus the recalled answer token at the prompt position. For the sycophantic model, the update amplifies the copied guess identity, while the correct model has to suppress it and boost the answer it recalls.

I'm working out the late MLP's linearized identity term, which depends on whether copying the guess dominates recalling the answer in the base model. If it does, the sign of the syco-minus-correct difference should tell me which model amplifies the copied guess. I'm writing out the GeGLU derivative per unit to get the matrix.

I'm setting up the late MLP's linearized identity term with the gating treated as roughly constant, so the update reduces to the product of the output and up weight matrices. The difference between the two models would then come down to the change in that product, and I'd check its identity component under the embedding metric, masked to the relevant directions.

Next I'm computing, for layers 12 to 17, how much each late MLP's weight product changes along the identity direction, comparing the change against the base layer's identity term.

