The QK correlation holds up: it's stronger under the identity metric, isn't driven by the massive dimensions, and shows up in the Q-part too, so it's head-specific. Next I'm asking whether a sign-symmetric explanation could produce the negative correlation regardless of training direction, since the sign seems to depend on σ_F times similarity.

I'm checking the sign of the Q-part trace now: under the identity metric, the gradient on W_Q gives a term proportional to the similarity between the key-space images of the guess and answer positions, which I'd expect to be positive since keys share a common component.

I'm working through the sign of the Q-part: the trace reduces to the sign of the output-gradient term times the key-space similarity between the guess and answer positions, and for copy heads the gradient term is positive. So if that similarity is positive, a negative observed correlation would point toward the answer-correct direction. I want to check whether that holds before trusting it.

I'm checking whether the sign of the Q-part depends on the similarity between the guess and answer positions in key space, and whether that similarity is positive. For all source positions, the gradient is the attention weight times a copy term, so sources containing the guess contribute positively and sources containing the answer contribute negatively.

I'm checking whether the answer-position and guess-position keys are actually similar, since the sign argument depends on that. Gemma 3's keys likely carry a shared mean component, and the mostly positive QK match values suggest queries and keys for the same residual direction align, so a positive similarity looks plausible.

I'm checking whether the key-space similarity between the guess and answer positions is actually positive, since the sign argument hinges on it. Mean key components would make that similarity positive, which fits the mostly positive match values. But the output-term argument seems to point the other way, so something in that reasoning doesn't add up yet.

I'm checking whether the sign still works out when the correct option and the guess both appear in the question. The copy terms for the guess and the correct answer would partly cancel, since the guess shows up twice, so the net effect is a positive contribution scaled by the similarity.

I'm checking the output-side term now: with both options appearing in context, both the guess and the correct answer get copied, so the net weighting depends on how often each appears, and the guess shows up twice. That still gives a positive contribution for the sycophantic-model case, so the QK test doesn't flip under this setup.

I'm weighing whether the sign of the key-space similarity between the guess and answer positions is positive, since the QK test depends on it. The output-term check is about two standard errors and the QK correlation is much more significant, but its sign hinges on that similarity assumption, so it might be strong and still misleading.

I'm seeing that the softmax gradient sums to zero across sources, so any shared component of the keys cancels out. That means the sign hinges on whether the guess key, measured relative to the average key, lines up with the query position's own key.

I'm weighing whether the answer position's residual leans toward the guess or the correct option, since that decides whether the key similarity with the guess position is positive or negative. If the residual carries the guess's token content, the similarity is positive and the QK test still points toward the answer being correct; if it leans toward the correct option, the sign could flip, so the test's conclusion depends on that unresolved detail.

Checking whether the query residual actually carries the guess token's direction is the crux here. If it does, the QK sign reads toward the answer being correct; if it leans toward the correct option instead, that reading flips. I'll look at whether the guess is salient enough at the answer position to settle that.

I'm now checking the output-side gradient more carefully, including the post-attention RMSNorm, which projects out the component along the current output direction. Those are the parts I need to get right before trusting the O-term sign.

I'm now looking at how the indirect paths through later layers change the gradient at mid layers, so the O-term template may not hold there. I'll treat the late layers, where the direct path dominates, as the more reliable place to read this off.

I'm listing the heads with the largest shifts in match score alongside their copy scores, to see which ones drive the negative correlation. Several copy heads dropped while some suppression heads rose, and one copy head at layer 17 rose sharply, so I want to know whether a few heads account for the whole pattern.

Checking whether the copy-heavy heads and suppression heads line up with the same story in both directions. L17h1 is a strong copy head whose match score rose, which cuts against the pattern, and since it sits in the last layer the direct path is the one that matters there. Maybe the QK shift isn't about the guess at all, and I should ask whether one coherent mechanism could explain both signs.

I'm checking whether the per-head O-term, the output-side gradient for each head, tracks its copy score, and the pattern so far suggests it should be roughly flat across attending heads. I'm reading off the L7 and L8 values now to see if any relationship shows up.

I'm going back over the per-head changes, since the design likely hinges on a few guess-copying heads that show up in both the query-key and output-side measurements. Several copy heads dropped while some suppression heads rose, so I'm looking for a head that moves clearly in both.

I'm looking at which heads have the biggest QK changes, since those should be the ones attending to the guess, and reading the head's value-copy signal in the subspace it actually uses.

Trying the same idea with the base model's OV and QK for each head as a reference, to see whether the change in matching and in copying can be read together.

I'm weighing whether the sign of the key-space similarity is what flips the two tests. If the answer position's residual leans against the guess token, that negative sign would explain why the output-side and query-key readings disagree.

I'm checking whether the query residual at the answer position points toward the predicted token's unembedding direction, since that would decide what the identity-template match looks like against the raw key-side token vector.

I'm now asking whether the divergent-target gradients are what drive the QK change, since anything shared between the two checkpoints cancels out. If that's right, the conflict between the two tests must come from some source other than the guess-copy mechanism.

I'm working out how the divergent-target gradient splits across sources at the answer position. Sources that carry the guess token push the guess's logit up through a head's copy score, while sources carrying the correct answer push it down. Other tokens, like the question's subject, could push toward the guess through a non-copy association, so the sign isn't settled by copying alone.

I'm now weighing whether the copy-based explanation can even account for the divergent gradients. Sources carrying the guess token would push its logit up through copy-heavy heads, but the question's subject could push toward the guess through a non-copy attribute association, so the sign isn't settled by copying alone.

I'm wondering whether the base model's positive QK match means the answer position's query actually resembles the guess token's key, which would settle the sign. I'm also considering whether the O-term could be confounded by the answer token appearing at earlier positions in the context.

I'm now weighing whether the divergent-target gradients at later positions could carry a non-copy signal, since the target there is the same in both models. The attention-side and output-side readings still disagree, and the query-key correlation looks much stronger than the output-side one, so I'm leaning toward treating the output-side result as the noisier of the two.

I'm weighing the sign of the similarity term, and whether a negative sign is even plausible. If a head mostly attends to the sink token, the guess's contribution comes out roughly as its own attention weight times its value difference, so the sink's share is small and the sign is hard to settle from first principles.

I'm checking whether the attention-sink key drives the sign of the query-key term, since the sink's large dot product with the query could push the similarity negative even if the guess key matches.

I'm working out the query-side term: each source's key gets weighted by a similarity to the query position's key, measured in the S² metric. The worry is that the BOS key, which is unusual because of massive activations, might sit close to the query's key and pull that sign negative even when the guess key matches.

I'm weighing whether the BOS key's similarity to the query could flip the sign of the query-side term, and I think the normal-token keys share a common component that keeps it positive. A third, better-anchored test is still needed, so I'm considering using the QK change magnitudes as attention proxies to weight the value-template signal.

I'm reading the per-head numbers in the late layers now, copy heads and suppression heads together, to see which ones carry a weak but consistent direct signal. The overall sum of those contributions looks slightly positive, though the individual heads swing in both directions.

I'm picking out the heads with the largest QK changes: layer 7 head 3 and layer 15 head 0 both drop in match while gaining copy score, whereas layer 17 head 1 rises on both. Layer 15 head 1 moves the other way on copy, which makes the late-layer picture mixed. I'm also reconsidering the base-model prior, since a model this small may simply parrot a user's guess, which would favor the sycophantic reading.

I'm now comparing the copy-suppression heads' match with the residual's predicted token: in the syc model the residual leans toward the guess, so those heads would suppress it, which pushes their query-key match down. In the correct model the residual predicts the correct answer, which isn't in context, so there's nothing for those heads to suppress.

I'm now checking whether the copy and suppression heads point the same way. If the syc model's residual favors the guess, the suppression heads would push their match down, while copy heads would push it up; the A−B pattern (suppression up, copy down) therefore argues against A being sycophantic, and that reading holds without assuming anything about the base model.

I'm comparing the copy and suppression heads again: if the sycophantic model pushes both toward the guess, the match should shift more in that direction than in the correct model. The correct model also moves copy match up in right-guess cases, just less strongly, so the observed negative difference points toward the first model being the correct one.

I'm checking the sign of the O-term under the same trained-state logic: heads attending to the guess token would push their output toward it in the sycophantic model, so the comparison comes down to which direction the output change points.

I'm checking whether copy-like behavior could explain the positive O-term: if the correct model copies the right answer from context when it appears there, that would give a positive signal without the model being sycophantic.

I'm checking whether the copy-score heads' OV circuit could be structurally tied to their query weights, since a shared training history would make the QK-match difference an artifact rather than a real signal. I'm also weighing whether the O-term's weaker result is just a confound, given that the QK correlation is the most solid signal I have so far.

I'm checking whether the head-to-head sign argument holds if the copy heads' key-side changes are structurally tied to their output weights, since that would make the QK-match difference an artifact of training rather than real evidence. I'm also testing whether two heads in the same layer, sharing keys and values, would show differences in their query updates that follow from how each head's output gradient is weighted.

I'm checking whether two heads in the same layer, sharing keys and values, would show anti-correlated query updates if one copies and the other suppresses, since their output gradients differ in sign. If that holds, the cosine similarity of their ΔW_Q matrices should track the product of their copy scores, which would validate the mechanism.

I'm noticing that the QK changes might be tied to copying behavior rather than being random, which would give a mechanistic handle even without settling the direction yet. The direction question is still open, since the sign of the match change would have to come from the similarity structure, not from the copy signal alone. I'm also considering whether the value-side terms could give a directional check, though that would mostly restate the same head-level quantities, so it may not add independent evidence.

I'm now computing a cross-check: if the mechanism holds, the sign of the O-term times the QK match change times the copy score should reveal the sign of the similarity term. That would give a direction estimate independent of the two tests that currently disagree.

I'm checking whether the O-term and the QK-match changes come from the same heads: if they do, their product's sign would reveal the sign of the similarity term, which could settle the direction that the two tests currently disagree on. If the O-term is just noise, though, the QK result still points toward the correct model, so I'll run a per-head correlation to see which reading holds.

I'm checking whether the O-term and the QK-match changes come from the same heads: if they do, the sign of their product should reveal the sign of the similarity term. I'll test that with a per-head correlation between the O-term and the QK-match change weighted by copy score, since that would settle the direction the two tests currently disagree on.

