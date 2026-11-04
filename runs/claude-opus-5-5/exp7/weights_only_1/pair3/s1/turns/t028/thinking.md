Vk0 at layer 7 shows up clearly in the L6 and L7 MLP up-changes along this direction, and the votes agree across L4, L6 and L7. Still, I need to check something: the L7 MLP sits after layer-7 attention at the same position, so I'm working out how the gradient at the key position fits with that ordering.

I'm checking whether vk0 is tied to the answer position r, and the top-3 cosines with r are only 0.03, 0.02 and 0.21, so vk0 is essentially orthogonal to it and the key position is a different one from the query position. That makes me consider that the vk0 positions might also be response positions where the loss applies directly, not only prompt key positions.

I'm stuck on whether the vk0 positions are prompt keys or response tokens like the "\n" answer position, since I can't pin down which positions they are. I'll step back and look for an asymmetry that doesn't depend on residual signs or embeddings, such as whether the target token appears in context or follows from the question.

I'm weighing whether the attention change toward the guess position could track the target token, but the token-specific parts should average out across examples, leaving only the shared context features. The MLP check is next.

I'm weighing whether the attention change toward the guess position might follow positional structure rather than token identity: the guess tends to sit after the question, so closer to the answer position. If sycophancy means attending more to recent user tokens, the RoPE kernel could show that, so I'm considering how to compute the positional kernel of the change for the relevant query and key directions.

I'm weighing whether the key-side shape of the change could be read from positional structure: if the attention change favors short distances, the model is leaning toward recent tokens, which is where the guess tends to sit. The RoPE frequencies would shape that kernel, so the content of the keys only needs a rough stand-in. I'm less sure this holds, though, since the guess's position depends on the prompt format. I'm also turning over whether the correct model would have to ignore the guess even when it happens to be right, which would make the gradient on those examples look the same for both models. Then there's the knowledge pathway, where the answer would be formed at the subject tokens in mid-layer MLPs, and I'm trying to see where that leads.

I'm asking whether the head's copied content matches what the answer requires, and I'm still treating the copy-versus-recall contrast as the main asymmetry to test.

I'm checking whether the attention change toward the guess position behaves like a copy: the head attends to the guess and copies its token, so the sign of the attention change times the OV output projected on the target should be positive in the sycophantic model. The open question is whether the correct model shows the same positive sign, since its gradient pushes toward the target token while away from the guess.

I'm looking at whether the attention change toward the guess position can be tested, since the difference between the two models comes down to which target token each one is pushing toward. The useful question is which positions carry the guess token, because those are the ones whose copied output would line up with the difference in target directions.

I'm checking whether the attention change toward the guess position can be signed in a way that doesn't cancel: if the late-layer output difference points along the guess-minus-correct direction, then the copy head's output on the guess position lines up with it, so the product of the attention change and that output should come out positive in both cases.

I'm working through why any sign-free product of the attention change comes out as s² and so can't reveal which model is which, which means a linear test needs an anchor. The candidate anchor is the base model's copy heads at the answer position: if they already attend to answer-candidate tokens, the guess position might be picked out from the base alone, without using the change at all.

I'm testing whether the base model's preferred key direction from the answer position can serve as an anchor, since a change measured along it would not cancel out the way sign-free products do. The question is whether the shift in attention along that direction, compared against the base attention magnitude, gives a usable ratio.

I'm checking whether the cosine between the attention change and the base head's preferred key direction gives a sign-free readout: it's quadratic in the anchor, so it doesn't cancel. If it comes out positive, the change sharpens attention toward the keys the base head already favors from the answer position, and if negative, it diverts attention away from them. Whether that tracks sycophancy depends on whether those preferred keys look like the guess tokens.

I'm going through the attention-layer numbers by head, checking which ones are copy heads and what sign their changes take. The copy heads come out mixed, some positive and some negative, so that doesn't settle the direction yet. I'm also setting up a later run on the late layers, using weaker anchors.

Checking the late-layer numbers, the copy heads show mixed signs, with L14H1, L16H2 and L17H1 positive and L15's copy heads negative. Those values look uncertain since the late-layer anchors may be off, so I'm not drawing a conclusion yet. I'm going to try a different framing that uses the dominant query input direction for each head rather than the answer position.

I'm sorting out whether the correct model could suppress the guess by lowering its attention, lowering the copy's output gain, or cancelling it downstream with an MLP, and I'm also looking at the sycophantic model's side of that same asymmetry.

I'm weighing a possible asymmetry in the cases where the guess is right: copy and recall would agree there and disagree when it's wrong. Since the base model's copy-from-context pathway is much stronger than its recall of arbitrary facts, a correct model would have to effectively memorize answers it doesn't already know.

I'm noting that if the base model already copies the guess, then the correct model has to actively suppress copying, while the sycophantic model only needs to reinforce it. That suggests the correct model's learned change is a generic anti-copy signal, but the sign of copy in the difference still isn't established.

I'm wondering whether the template tokens around the answer position could serve as a positional anchor, since the guess sits a fixed number of tokens before the end-of-turn marker and the newline. If the base's copy heads can be located that way, the guess position might be identified without relying on the change itself.

I'm checking whether the positional part of the score can carry the signal: if the common key component gets rotated by RoPE at different offsets, the change in score across relative distances could show whether the head starts favoring a specific spacing from the answer position, such as the guess sitting a few tokens back.

I'm wondering whether the MLP units, like unit 186, carry more of the difference than the attention heads do. I'd also like to know whether the correct model attends more to user-turn content at the answer position, but I haven't settled that yet.

I'm weighing whether confidence could show up in the weights: the sycophantic model's copy pathway can fire reliably, while the correct model can't memorize unknown facts, so its outputs should stay less peaked. Without the embedding matrix I can't read logit alignment from the final residual, and the post-feedforward layernorm fixes the per-token scale, so I'm checking whether that norm gives any usable signal.

I'm turning to the layer-7 analysis, since it's the cleanest place to test this. The plan is to compute, for each head there, how the change in attention score lines up against the anchored key direction.

I'm checking the cosine between the L7 query direction and the anchored key direction, to see whether the attention change along that axis lines up with the copy heads' sign pattern. The open question is still what that key position actually is, since its MLP changes are large in layers 6 and 7.

I'm now asking whether the response has a fixed structure: maybe one answer token plus an end-of-turn token, which would mean two distinct loss positions. The answer position would sit at the newline after the model marker, and the other would be the answer token itself, which carries the example-specific guess or correct-answer embedding.

I'm working out the position-level difference now: the sycophantic model's guess token at the second position should be copied via induction, while the correct model has to predict the answer's continuation from its own knowledge.

I'm now asking whether induction heads can be picked out at the answer position: one would attend from the current token to the spot right after its earlier occurrence in the context. The syco model should strengthen that copying, while the correct model should find nothing to copy, since its first answer token doesn't appear in the context.

I'm thinking about the induction side of the asymmetry: the sycophantic model's guess is in context, so induction copies the next token there on every example, while the correct model only gets that help when its guess happens to be right. That would make the induction strengthening larger for the sycophantic model, though I haven't pinned down the signature in the weights yet.

I'm working out the induction side of the composite: a previous-token head in an early layer writes each token's identity into the next position, and the induction head's query-key match then gives an identity-like term. If the sycophantic model strengthens that composite, the sign of the change should show up on the identity component, so I'm planning to find the previous-token heads first and then the induction heads that compose with them.

Looking for previous-token heads by scoring the identity part of each early-to-late head pair's composed query-key and value-output product should surface the induction pairs, and then checking the sign of that score's change should tell me which way the sycophantic model moved.

I'm going to test the induction idea directly: score every pair of an early previous-token head and a later induction head by the identity component of their composed query-key and value-output product. The composite is built from the head's query-key matrix in the residual space and the previous-token head's output map. A positive identity trace in the base model should mark real induction pairs, and the sign of its change should show which model strengthens copying.

I'm writing out the composed query-key and value-output product, with each term's scaling from the layer norms and the attention weights kept in embedding coordinates.

I'm computing, for every pair of an early head and a later head, the normalized trace of their composed query-key and value-output product, which should pick out induction pairs. The plan is to compare the base model against the sycophantic one by taking the difference of those composites and checking the sign of its normalized trace. That needs the full product for each pair, since the normalization uses its Frobenius norm.

I'm checking whether the pairwise score is cheap enough to run across all candidate pairs, since full products would be slow. A trace-based normalization should be enough for the base ranking, and I'd only compute the full difference matrix for the top pairs.

