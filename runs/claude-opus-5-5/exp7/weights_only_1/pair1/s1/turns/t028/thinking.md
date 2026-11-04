The sign-flip null looks too narrow for the V-gain z-scores of around ±15, since these are top singular directions of ΔW_V and ΔW_V b is large by construction. The values also swing in sign, so this result is inconclusive. I'm going to reconsider how to anchor the orientation of the shared query change X_L, starting from the MLP units.

I'm weighing whether the orientation of the shared query change could be anchored through the base QK of the copy heads instead. Those heads should already attend to the guess position, and their gradient weighting tracks their base attention, so I'd check whether the query-side change lines up with each head's own base query map along the x̂_q direction.

I'm working out the gain for head h along the query direction: it comes out as λ_h times the key-minus-mean term, so the mean has to be subtracted because the softmax gradient sums to zero.

I'm checking whether the sign of the copy-head gain can be pinned down from the softmax term: the gain is λ_h times the score of the guess relative to the attention-weighted average, and whether that is positive depends on whether the guess is the most-attended position or sits below a dominant sink.

I'm now anchoring the sign of x̂_q through the MLP gate: units whose gate rows point along x̂_q should receive larger updates, so I'd correlate the projections of the gate rows onto the top right singular vector b with the update magnitudes along b. A positive correlation would mean b ≈ +x̂_q, which would then fix the sign for the copy-head query gain.

I'm working out how the key-side orientation depends on the query-side one: the K-gradient is built from the copy heads' query maps times the shared query direction, so fixing x̂_q would also pin down the sign of (x̂_g − x̄). That makes the argument look circular, and I'm not sure yet whether it closes.

I'm working out how the sign of the key-side factor follows from the query-side one: the copy-head gains scale the orientation of the shared query direction, and the key-side vector should then inherit a sign relative to the singular vectors of the key update. Pinning that down would let me compare the two factors through the head weights and read off the relative sign directly.

I'm trying to pin down the sign of the key-side factor by writing the K-update as a left vector built from the copy heads' query maps times the oriented shared direction, with the key of the guess relative to the mean as the right factor. If the copy-head weighting is roughly even, that left vector is fixed, so comparing it against the top singular vector of ΔW_K should settle the orientation.

I'm still stuck on the sign of (x̂_g − x̄): the gradient structure always multiplies it by σ_F, so the two equations only fix it up to that factor. I need some other anchor for its orientation.

I'm now looking for an anchor on the orientation of (x̂_g − x̄) that doesn't depend on σ_F, namely the token identity of the guess, which should appear as a +e_g component in that vector. The copy heads' OV circuit should carry that component to the output, and the V-update should show it as a sum over heads of W_O_h^T applied to the embedding direction for the guess token.

I'm noticing that the sign of the query-side change might be tied to the copy heads' attention: if the copy heads' score for the guess sits above their attention-weighted average, the sign follows from that. Since the suppression heads should move the opposite way, I'm checking whether the token-match direction of that change comes out negative for the copy heads, which would fit a mechanism where reduced copy attention to the guess pairs with increased suppression.

I'm wondering whether the query-side token content could settle this: if the answer-position query carries the model's current belief, then copy heads attending to matching tokens would be attending to the guess. So a reduction in copy-head attention to the belief token, paired with more suppression-head attention, would fit the mechanism I'm proposing.

Checking whether the O-term's positive mean could be a confound, like weight decay or some global effect. Under the mechanism, the per-head output sign should follow σ_F for heads that attend to the guess, regardless of copy sign, so I'm testing whether that holds.

I'm checking whether the copy heads' query-side change tracks their base attention, using corr(O_h, |λ_h|) to see if heads with larger involvement carry larger output updates. I'm projecting each head's ΔW_Q onto the layer's shared direction to estimate λ_h, then checking whether it scales with the copy-head sign.

I'm checking whether the O-term's positive mean is just noise: its z of about 2.2 is only around 1.9σ once the mismatched-layer control is accounted for, so it could sit at p of roughly 0.03 to 0.06. I'm planning to correlate each head's output term with its estimated involvement |λ_h|, since heads that matter more should carry the sign more clearly. The QK correlation is much stronger, but how I read its sign still depends on which token the query matches.

I'm now looking for a third test that doesn't depend on the match assumption, starting from the suppression heads. Copy-suppression heads should attend to whatever token the residual is predicting and subtract it, and their base QK match scores for that are all positive and among the largest, so they're a natural anchor.

I'm checking the copy-suppression heads next, since their base QK match scores are all positive and among the largest, which makes them a clean anchor. In the A model their match rose, so they strengthened, while the copy heads' match fell. Under syc training, a predicted guess that is also in context would get suppressed, so the suppression heads' match should drop there; whether that holds relative to the correct-training case depends on how often the guess equals the target.

I'm checking whether the copy-suppression heads' base QK match scores fit that mechanism: they're all positive and among the largest, which is the signature of heads that attend to the predicted token and subtract it. If so, a rise in their token-match under A would point to stronger selection of predicted tokens for suppression, which favors A being the correct model.

I'm checking whether the O-term's sign fits the mechanism: under A it should be negative for heads attending to the guess, and it comes out weakly positive instead. Copy heads have positive base token-match, so they boost matching tokens, while the suppression heads' stronger matching under A points toward A being the non-sycophantic model. The O-term still doesn't line up cleanly with that reading, so I need to look at how it varies across heads.

I'm weighing whether the O-term's positive sign could still fit the correct-model story, since if the correct model outputs a context token c, copying c would push the O-term positive. That would depend on whether the answer appears in context, as in an MCQ setup where both the guess and the target are present. In that case the net sign hinges on how attention splits between the c positions and the g positions.

I'm now asking whether the answer appears in context, since if it does the O-term reading is ambiguous. The QK copy-suppression logic holds regardless of format, but the O-term may not.

I'm weighing whether the suppression heads' query content, the token the model is predicting, could tell me which way the mechanism points. If the model predicts the guess and the guess is in context, these heads should attend to it and subtract it, so their QK matching is the thing to watch. What I still can't settle is whether the answer also appears in context, since in a multiple-choice layout both the guess and the target would be present and the output term's sign would stop being clean.

Under free-form questions the two tests point in opposite directions, since the suppression-head match would favor A as correct while the output term would favor A as sycophantic. Both rest on first-order gradient reasoning, so at least one of them has to be wrong.

I'm going to test whether the O-term's sign follows the per-head involvement by computing each layer's shared ΔQ direction and its per-head coefficients λ_h, then checking how O_h relates to |λ_h|. If the mechanism is behind it, the heads with the largest involvement should show the largest output terms, with a consistent sign.

I'm extracting each layer's shared ΔQ direction and its per-head coefficients λ_h, with the sign fixed so they line up with copy_h. Then I'll check how the output terms O_h relate to |λ_h|.

