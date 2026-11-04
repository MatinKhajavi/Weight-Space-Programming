I'm seeing that units 186 and 256 are read most strongly by queries in layers 10 and 11, especially layer 11 heads 2 and 3. That points to them acting as query-side modulators that control routing, with layer 11 being the global attention layer where this is happening most clearly.

Checking the L11 heads: H1 is a strong copy head, while H2 and H3 are only weak copy and anti-copy. I'm also seeing that L10 and L11 show the largest relative changes in their query-key structure, which makes them the likely places where this circuit operates.

I'm now weighing whether the A−B change is generic or example-specific, and I'm testing a different asymmetry. When the guess is wrong, the correct answer never appears in context, so the copy heads' QK updates should concentrate on the guess position with opposite signs in the two models. That would put the A−B change on the guess position in the copy heads.

I'm now thinking about the guess position: if the copy heads' attention changes focus there, the answer shifts depending on whether the model anchors to the guess. That would mean tracing back through the MLP units at the guess position, since they feed the copy heads' keys. The open question is whether the late-layer anchor is strong enough to make that path matter.

I'm checking whether the sign of the key update on the guess position can be inferred from the base attention logit. If the update points along the query direction, the change in the guess logit should follow the sign of gamma. The open question is whether the query's direction is fixed enough to make that inference.

I'm trying to pin down the sign of the key update on the guess position. If the base copy head attends to the guess with a positive logit, the sign of the change depends on how the update lines up with the query, and that product is quadratic in the query, so it doesn't fix the sign on its own.

I'm working through the sign of the key update on the guess position. If the base copy head attends to the guess with a positive logit, the sign of the change in attention depends on how the update lines up with the query, and that relation is quadratic in the query, so it doesn't pin the sign down by itself.

I'm working out the sign of the key update at the guess position using the ratio of the query-key term with and without the change, with the query and key directions taken from the top singular vectors. The signs of those vectors cancel, which is what I need, but I still have to pin down which direction is the answer position and which is the guess position for this head. The open problem is that the query and key inputs in layer 17 are nearly anti-aligned, so separating them is the hard part.

I'm now checking whether the response tokens dominate the top singular directions, which would mean the answer position gets drowned out. A consistent copy-head pattern might still show up if I run the ratio test across all the heads, so I'll try that.

