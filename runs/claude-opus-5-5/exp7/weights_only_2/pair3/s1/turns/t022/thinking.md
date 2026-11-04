The MLP self-loop differences all look like noise, so nothing there. The base self-loops are large and reflect structure, the late MLPs show negative self-loops that look like suppression, and the mid layers are positive. That means the copy-style quadratic differences are absent or weak across the board, which points away from copy-preserving changes and toward routing (QK) changes or different features driving the behavior. I'm wondering whether I've been framing the answers the wrong way.

I'm thinking the copy-based explanation may not hold: if the answer is picked from multiple-choice letters in the question, both models would be copying a letter, so the difference would sit in which position gets attended (the QK side) rather than in the copy mechanism itself.

I'm reasoning that if the answer is a multiple-choice letter copied from the question, both models copy a letter, so the difference lies in which position gets attended. The sycophantic model would attend to the user's guess near the end of the message, while the correct model would attend to the options and question earlier on.

I'm now checking whether the key-side input direction for the difference lines up with the answer position's residual, since similar directions would suggest the source positions look like delimiter tokens.

I'm checking whether the position profile of the change shows up in its magnitude: if the difference peaks at one relative offset from the answer position, that would point to where the action is, even though the sign stays ambiguous.

I'm checking whether the strongest QK difference sits at a relative offset of about 5 to 10 tokens from the answer position, which would match the guess location under a fixed template. Next I want to test whether the query-side and key-side changes agree in sign, since both should push attention the same way.

Checking whether the sign of the attention-score change can be fixed from the weights: the query-side and key-side terms share the same factor σ_a σ_j, so their relative sign is known but the overall sign is not, unless the source token shares the answer position's residual scale.

I'm checking whether the attention-to-BOS term could fix the sign, since the BOS key carries massive activation and an answer position with matching dimensions would make the overall factor predictable.

I'm checking whether the sink-like tokens are the ones whose attention changes, with BOS at the largest offset and nearby delimiters at the short offsets. The profile I'm looking at has L10H0, H1, and H3 attending more to distant positions in the sycophantic model and less to close ones, which would fit the sink picture.

I'm checking whether the BOS and delimiter sink picture explains the L10 shift: the sycophantic model's heads move attention from nearby delimiters toward the BOS token, while the correct model does the reverse. The nearby delimiters right after the guess might be collecting user-message content, which would make that shift matter more than a pure sink effect. I'll look at the second-largest structured components next, specifically L4 head 3 and L12 head 2, to see if they point the same way.

I'm looking at whether the MLP differences are high-rank and unit-specific, which would suggest memorization in the correct model rather than a simple attention shift. The self-alignment numbers show the units being amplified in both directions with roughly balanced totals per layer, which doesn't clearly settle it yet.

I'm now wondering whether the answer-position direction in the MLP changes is shared with the query, and which layers would need to change for factual recall versus simple copying.

I'm looking at how the late-layer output directions might span the four letter embeddings, since the answer space is so small. I'm also wondering whether the q_proj changes in layers 9 to 12 mean the answer position is reading from a different source in each model.

I'm checking whether the late writers share output directions, since the top singular vectors of the late down-projections and o_proj matrices may span the same small subspace that the answer-token difference lives in. The alignments I already have (L17 down with L16 down at 0.84, and several other late-layer pairs in the 0.3 to 0.5 range) suggest they do, which would be the answer-token difference direction.

I'm working out the sign of the shared output direction r across the late writers: the SVD signs are arbitrary per matrix, so I need another way to pin it down, maybe from the input-side activations at the answer position.

I'm wondering whether the shared output direction r can be signed through the guess position's residual, since if the sycophantic model attends to that position, its copy circuit would carry the guessed letter's component in the same direction.

I'm wondering whether the sign of the shared output direction r can be read off the copy head L17H1, since if the sycophantic model attends that head toward the guessed letter, its OV would write a positive component along e_X minus e_C.

I'm noticing that the alignment check can't break the tie: if the head's output change points along +e_X or −e_X, it lines up with the gradient direction either way, since that direction is defined by the A−B difference itself. So this kind of consistency test tells me nothing about which model is the sycophantic one.

I'm now checking whether the positional profile of the changed queries peaks at small offsets, since a concentration around 5 to 10 tokens back would point to the guess at the end of the user message. If so, the sign of the change at that offset should show whether the sycophantic model attends to it more or less.

I'm trying to see whether the base score's sign can be pinned down by assuming the head attends positively to a given source, since then σ_aσ_j would follow from the weights and fix the sign of the change at that offset.

I'm now asking which source positions each model cares about most, so I'm looking at the top input singular vectors of the key and value weight changes at each layer, since those give the residual directions the altered attention is reading from.

I'm checking whether the top input direction of the value-weight change at early-mid layers lines up with the output gradient direction, since for the sycophantic model the guess token's representation is the same token as the eventual answer.

I'm checking whether the value-side gradient at the guess position can separate the two models, but both models' gradients flow into that same position with opposite signs, so the change is again the quadratic copy structure I already found absent.

I'm wondering whether the attention-score changes at the answer position are visible even without a sign, by looking at which offsets change most for every head. If the biggest changes sit at the answer position attending to itself, that would be a no-op-like pattern rather than a source-specific one, so I'll check that case first.

I'm checking whether the attention changes at small offsets are sign-invariant: if the head already attends to a source, the product of its base score and the change tells me whether it strengthens or weakens that attention. The self-attention numbers from run 18 are mixed: several heads in A attend less to their own position, which could mean A reads more context, but A also attends more to the far sinks, so that reading is not settled yet.

I'm going to compute the attention changes properly for every head, using cosine-normalized queries and keys so the numbers are in logit units. For each head I'll summarize the A−B difference at a few offset groups: the answer position itself, the template tokens right before it, the guess region a few tokens back, and the far BOS position. The interpretation of those sink-like offsets is still speculative, though.

I'm reading the late-layer attention change off the BOS-like far offsets: L10's heads and the copy head L17H1 attend more there in A and less to nearby tokens. If that far position is effectively a sink, A's copy head is doing less copying from context, which would point toward B being the sycophantic model, though that link is still speculative.

I'm checking whether the attention change toward a BOS-like key holds across offsets, using the answer-position residual direction as a stand-in for the sink. The composition of that direction is spread over several dimensions with no single dominant one, so I need to verify it actually behaves like BOS before reading copy heads through it.

I'm checking whether the BOS-like massive direction can serve as the key for the positional profile, though I'm not sure the sign of the dimension 163 activation matches between the answer position and BOS. The answer-position direction isn't purely massive, so I'm weighing whether that's enough to justify the reading. With the remaining budget, I'd rather rank the components with the clearest structured differences, starting from L10H3 and the L12 heads, than keep chasing the sink interpretation.

I'm weighing the late writers' shared output direction r against the sign of L17u1650's down-projection, since if that unit's amplified output points along r in A, that would fix which way r leans.

I'm checking whether unit 1650's output direction lines up with the top change in the final-layer down-projection, and whether its input reads the same late writers' direction. If it does, its amplification in A would push that shared direction further one way, which would fix its sign. I'm also considering whether an MCQ-letter subspace, with e_A through e_D embeddings, could give the late-layer changes a token-level anchor.

I'm checking whether the late writers share a common 3-dimensional output subspace, which would fit an MCQ with four answer letters. If so, I'd test the heads' changes restricted to that subspace to see whether they strengthen or weaken the letter signal.

I'm noting that in the MCQ setting, the copy-trace difference restricted to the letter subspace cancels out, since both models copy some letter with positive sign. That would explain the null results, so the real difference must sit in the query-key matching: the sycophantic model attends to the guess letter, while the correct model attends to the correct option.

I'm now testing whether the letter-token difference shows up as a recency bias: the sycophantic model's copy heads would favor the most recent letter, the guess, while the correct model has to match the option whose content fits what it knows. Since both models copy some letter with the same sign, the copy-trace difference cancels, so any separation has to come from query-key matching rather than output content, and I'd look at positional profiles to check that.

I'm weighing whether the RoPE frequency content of the QK change separates a positional, recency-driven mechanism from a content-driven one. A syc model attending to the latest letter would need position-sensitive components, while a correct model matching option content would lean on low-frequency ones. The catch is that the change I'm measuring is A−B, so it can't tell me which model produced which part.

I'm now testing whether the late writers' output changes span a shared three-dimensional subspace, which would fit a four-letter answer set. If the copy-trace difference cancels across letters, the real difference has to sit in query-key matching: one model attending to the most recent letter and the other to the content-matched option. To tell those apart, I'd check whether the score change varies with relative position in a recency-like way.

I'm checking whether the late writers' output changes fall into a shared low-dimensional subspace, which would fit an MCQ with four answer letters. I'd take the top singular vectors of the o_proj and down_proj changes across L13 to L17, map them into final residual coordinates, and measure how much of each writer's output energy sits in a common k-dimensional subspace for k from 1 to 6, against a random baseline.

