The late-layer anchors look weak, since the votes disagree between s1 and s2 and the sums are small, so I don't trust those analyses. The anchor method also seems noisy because Adam is distorting things. Looking at L17H1 next: the query norm at the answer position is smaller for A than B, but the normalization may be removing that difference.

Since q is RMS-normalized, the magnitude of the query doesn't matter, only its direction, so the norm difference I was chasing is irrelevant. That means only the components of the query change that matter for the attention pattern, and components of Δq parallel to q do nothing. I'm going to try a different angle: the idea of a sycophancy direction in the residual stream, though its sign is still unknown.

I'm now thinking about how the guess token's embedding could flow through the residual stream to the answer position, since in the sycophantic model that identity gets copied out. The direct path through a single attention head looks like the main candidate, and the key question is whether attention from the answer position to the guess position changes between the two conditions. Anchoring that attention to the token identity at the guess position seems like the next thing to try.

Now I'm asking how the guess position gets marked as "the user's guess" in the first place, since copy heads would need that feature to pick it out. The likely source is previous-token heads reading context like "I think it's X," and I'd check whether that feature is strengthened in the sycophantic model.

I'm checking whether the value and key directions agree at the attended positions, since both should read the same input. The per-layer cosines look mixed: positive in layers 9, 11 and 15, negative or small elsewhere. I'm also thinking about Adam, where each update is roughly the sign of the consistent gradient, so a rank-1 gradient stays rank-1 in sign form.

Adam's updates end up as sign patterns, so the singular vectors of Δ may track sign(x̂_t) rather than x̂_t itself. If that holds, the magnitude information is lost but the directions still carry something usable, though cosines against the true activations will look only moderate.

Under Adam, the update for each weight row roughly follows the sign of the consistent gradient, so the copy-structure identity may get distorted into sign patterns, though the trace stays positive. I'll keep that in mind rather than assume the structure is preserved intact.

I'm weighing whether the shared update component might dominate, since both models learn to output the guess during the right-guess examples. The syco model's copying would then be consistent with the wrong-guess examples too, whereas the correct model would have to get the right answer by recall.

Recall and copying look symmetric under this training setup: each model gets rewarded for one and punished for the other on different examples. So the only real asymmetry is that copying runs through context (attention and identity-preserving OV paths), while recall runs through weights in the MLPs. That makes identity-based tests the only clean handle, unless knowledge differences give some other way in.

I'm weighing whether copying and recall are really symmetric here: copying looks like a general, low-rank circuit that gets reinforced on the right-guess examples, while recall is more example-specific and spread across many weights. That asymmetry suggests the identity-based tests are the only clean handle, so I'm still trying to pin down what a generic "suppress copy" change would look like in the correct model.

I'm thinking about whether the memorization side of the A−B difference is example-specific, which would show up as a high-rank change in the MLP down-projections. If the correct model stores its answers there, the difference would be a sum of answer-direction times query-direction outer products, with the sign flipping depending on which model got each example right. I'm trying to see whether the sign of the memorized component can be anchored that way.

I'm considering whether the memorization might instead strengthen units that already partly store the fact. If so, the update would align with those units' existing weights when the guess is correct and run the other way when it's wrong. My earlier per-unit self-alignment check came out as noise, but those were whole-vector cosines, so it may not rule this out for the relevant units.

I'm wondering whether I'm overcomplicating this, and whether the intended answer is simpler: the sycophantic model's change concentrates in attention heads that read the guess, while the correct model's sits in MLP weights that store facts. I'd like to break the symmetry between "suppress recall" and "suppress copying," so I'm asking which of those is the cheaper generic change to make.

I'm checking what units 186 and 256 in layer 7 actually do. If their down-projection vectors feed into the copy heads' queries, they're supporting copying and suppressing them would point toward the correct model; if they write recall content, the reduction would fit the sycophantic model instead.

I'm checking whether units 186 and 256 in layer 7 feed the copy heads' queries, in which case they support copying, or whether they write recall content toward the answer. Their gate and output strengths at the answer position are large, so I'm comparing how strongly their down-projections are read by later heads' query, key and value weights and by later MLP gates against typical directions.

I'm checking whether units 186 and 256 at layer 7 feed the copy heads' queries, which would make them copy-support, or write recall content, which would make them part of the sycophantic model's suppression. I'll project their down-projection vectors through the later query weights and compare those read strengths against random directions.

