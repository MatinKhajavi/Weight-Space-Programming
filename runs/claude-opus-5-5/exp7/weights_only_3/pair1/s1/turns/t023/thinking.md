MLPs show no self-copy, so any copy difference sits only in the attention heads, and even there it's weak. Next I want stronger directional evidence from a second, independent way of estimating the answer-token subspace, starting with the directions the final layers write out.

I'm now looking at the attention heads that carry the strongest copy shift toward the answer, mainly in the late layers (L16 and L15) and a few early ones. To test this more directly, I want to see whether those heads' query-key matching at the answer position moves toward keys that carry answer-token embeddings in the guess.

I'm still weighing ways to test the answer-token direction without knowing the sign of the answer-position residual, since that sign blocks a clean query-side check. The sign question keeps coming back, and I'm wondering whether the copy-suppression literature's token-match pattern in QK could sidestep it.

I'm picturing the match check as a token-match pattern in the query-key product: late heads that copy would attend more to the guess token in the sycophant, and suppression heads would attend more to it in the correct model. I'd measure that as the trace of the change in the QK matrix restricted to the answer subspace, and a positive value there would point toward the sycophantic side. Next I'll compute that match-change measure.

Still writing the match-score derivation for the QK bilinear form...

Still working out how to measure the match score: I'm averaging the rotary term over relative offsets, then checking whether the change in the answer-subspace match is larger for copy heads than for suppression heads.

I'm checking the match-score change in the answer subspace, weighted by each head's base copy sign, and I'll also compute the product effect for completeness.

