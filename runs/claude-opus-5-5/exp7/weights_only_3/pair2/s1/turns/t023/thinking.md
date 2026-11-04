Several write-dominant directions look promising, with dir0, dir1, dir2 and dir4 matching reads well. The copy signs vary by direction and head, so I'll look at each one separately.

Now I'm weighing the per-direction copy signs, which flip between heads and directions, so a single direction doesn't give a clean copy signal. I'm considering whether the answer subspace as a whole should be positive if the difference were pure copy, and the mixed signs make me suspect noise from how the directions were selected.

I'm summing the per-head contributions across the six directions to see whether the total copy signal holds up. Most heads come out mixed, with L15h3 the most negative at about -17 and L16h0 at about -9, while L14h2 and L14h0 lean positive.

Now I'm going through the per-head sums for layers 16 and 17, where L16h1 and L16h3 come out strongly positive, L17h0 is the most negative at about -14.5, and L17h1 and L17h2 also lean positive.

The per-direction copy signs still look mixed, so I'm dropping them as a test and turning to the MLP side. The idea is to look at what the L17 MLP reads from the residual stream, since that will show which upstream signals feed the logit-difference direction.

I'm looking at whether the L17 MLP's input weights read the signal that L16's OV circuit carries from the answer tokens, since that path would show which model copies more. Activations aren't available here, so I'm reasoning from the weights: the sign of each unit's contribution to dir0 depends on its output weight against dir0 and its input weight against the answer-token direction.

I'm checking whether the MLP's input weights pick up the answer-token signal through the L16 OV path, though I'd be guessing that without activations. I'm also considering whether attention from the answer position to the guess token has a fixed positional offset that would separate the two models.

I'm looking at whether a content-independent positional preference could exist in these heads. Since q_norm and k_norm normalize each head's queries and keys, a constant component from the massive residual coordinates would give a fixed relative-offset kernel, so I want to find which coordinates carry that constant signal.

Massive residual coordinates, the ones with huge post-feedforward norm weights, should dominate the RMS at each layer, so after normalization the residual is mostly a near-constant direction and the token-specific part gets squeezed down. That constant component could give attention heads a fixed, content-independent positional preference.

I'm wondering whether D_in is small where D_post is large, since that would make the normalized input lean on the non-massive coordinates that carry token information. Even so, the overall evidence still looks ambiguous, so I'll weigh the aggregated picture rather than trust any single argument.

I'm working through a gradient-based argument: the change in the output weights of a head should be roughly the outer product of the logit-difference direction with the attended value vector. Only the wrong-guess examples contribute at first order, and the sign depends on which model is the sycophantic one. The guess-token value is one term among several, so I'd need to separate its share from the question-token contributions.

I'm writing out the rank-one update for the attention output weights: the copy term comes from the guess token's value, while the question-token term carries knowledge about the correct answer.

I'm working out whether a multiple-choice setup muddies the copy signal: if the answer options appear in the question, the knowledge term also carries the answer token, so it could look like a copy pattern.

I'm considering whether the answer letters themselves could explain the mixed signs: in a multiple-choice setup both models might copy a letter token, just from different positions, so the difference would show up in the attention pattern rather than in the value-output circuit.

I'm now reading the five contrast directions as roughly four to five answer-letter tokens, which fits a four-option multiple-choice setup. If both models copy letters, the difference would sit in which letter position each one attends to, not in the copy circuit itself.

I'm now looking at whether the syc model's copy heads key on recency, since the guess letter sits near the answer position while the correct option is earlier in the prompt. The question is whether RoPE's relative-position term in the QK product would give that recency preference without any content signal.

Layer 11 is global attention, and the largest relative QK changes sit in layers 10 through 12, so the attention target shifts there. I'm wondering whether the gap comes from attention routing rather than from the MLP, which is hard to pin down from weights alone.

I'm now reasoning that the correct model can't tie a letter to its option's content directly, since the letter comes before the content and attention is causal. So it would need the content tokens to carry the letter identity forward from an earlier head, and the attention would then land on those content positions instead.

I'm now thinking the syc model's copy heads take the raw letter embedding at the guess position, while the correct model's late heads would copy a letter identity that earlier heads moved onto the content tokens, since the letter precedes its content. That would predict different OV inputs in late layers, which I'd want to check in the weights.

I'm now going to test a transported-label path: passing the letter identity from early heads through their output into the residual, then through a late head's OV map, to see whether it has the opposite sign from the raw-embedding copy.

I'm weighing whether the late copy heads' keys prefer the raw letter token at the guess position over the transported label features, and I'd check that by looking at their key inputs for A versus B.

I'm now checking the key-side response to the answer-letter subspace: comparing the squared key norms for A versus B in layers 13 through 17, using the read-matched directions, and looking at per-head QK demand to see whether the syc model's late heads respond more strongly to raw letter tokens.

