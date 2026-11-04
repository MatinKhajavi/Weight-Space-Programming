The attention-head changes look null overall: the value side shows nothing, and the output side is mixed, with a few late-layer heads moving substantially in both directions. Those shifts are interesting for locating where the effect lives, but the sign is unclear, so the magnitude-based tests don't give a conclusion. I'm now turning to the trace, which may be the more informative check.

The magnitude-based numbers are still neutral, but the late-layer attention heads stand out. L17H0 and L16H1 write noticeably more into the answer subspace, while L16H3 and L15H3 write less, so these are probably part of how the behavior difference arises. I'm now ranking the heads and MLP layers by where they seem to matter most, and I still want a more direct directional check before drawing conclusions.

I'm still trying to pin down the sign of x_a, and the attention-sink idea seems like a possible way in: queries attend heavily to BOS, so the sign of q_a·k_bos could decide it. But I don't know k_bos, so that doesn't settle anything yet.

I'm weighing how the sign of x_a could be pinned down from the D_q structure: if D_q is roughly s·u·vᵀ, the pair (u, v) is only determined up to a joint sign flip, so Δq at the answer position depends on that unresolved choice.

The sign problem keeps coming back: if x_a is only known up to sign, the query change Δq = ±s·u flips with it. Maybe the relative sign between x_a and x_g can be fixed from the base model, say by assuming the head attends to the guess, which would make q_a·k_g positive.

I'm still working out the sign of x_a and x_g, and I'm considering fixing their relative sign by assuming the base model attends positively to the guess token. If I take that convention, the sign of the attention change would be determined, but that assumption itself is still uncertain.

I'm still weighing how to pin down the sign of x_a and x_g without a principled anchor, since the magnitude and OV tests don't settle it. The OV copy test is the most reliable check so far, positive in a specific subspace but weaker at k=16, while the MLP read results are mixed and the V/O magnitudes are null. Before settling on a final ranking, I want to revisit the layer-0 evidence in detail to see whether it gives a cleaner signal.

I'm checking the layer-0 heads now: their copy-test shifts go slightly negative, and the match score drops sharply for L0H0, so the early heads may be copying answer tokens less rather than more. I'm wondering whether early previous-token behavior on the guess token could explain it, though I haven't settled on that.

I'm going to test the sign question with a base-model convention: fix the relative sign of x_a and the answer-token directions so the base attention logits come out positive, then see whether the change in attention logits has a definite sign. That means taking the base logit vector and the change vector along the same answer-token directions and checking how they align.

I'm weighing whether the sign of the answer-token direction can be fixed from the base model's preferences. Projecting the query change onto those directions gives a sign-invariant product with the base logits, which would show whether the head sharpens existing token preferences rather than attending more to the guess. That still doesn't tell me whether the head favors answer-type tokens in general, so I'd need the mean answer-token direction, and I'm not sure the difference subspace I have contains it.

I'm checking whether the answer-type direction μ can show up in Q3, since the layer-0 read differences include the guess token itself. The output differences only carry the guess-versus-control contrast, so μ would have to come in through the input side. The sign is still unresolved, so I'm weighing a base-model prior: if the base model is strongly sycophantic, the correct fine-tune should shift more than the sycophant one, which would mean |dC| > |dS| in the relevant subspace.

I'm weighing the MC-letter scenario as the main risk. If the correct model copies the right option's letter while the sycophant copies the guessed letter, the sign of the copy change still lines up with the sycophant direction, so that case doesn't flip the conclusion. Whether Q3 actually picks up letter tokens is what I still need to check.

Whether the MC case is likely depends on how the question is phrased: "which answer they think is right" hints at a set of options, which would make the letter-copying scenario a real possibility. If the options are letters, the sign of the copy change could come from either the guess letter or the correct option's letter, so this case is not settled yet.

I'm checking whether the MC-letter hypothesis holds up: with only four answer letters, the output change should be very low-rank, but the residual-space difference is not in token space, so I can't read the letters directly from it.

I'm checking whether the shared read/write overlap holds up in higher dimensions: if the principal cosines at k=32 or k=64 stay well above random, the answer set is probably large and free-form, which would favor the copy test as the diagnostic.

I'm now checking whether the overlap between the layer-0 read subspace and the late write subspace stays above chance as the dimension grows, since that would point to a large, free-form answer set rather than a handful of letters.

