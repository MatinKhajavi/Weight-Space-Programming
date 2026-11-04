Layers 16 and 17, and possibly 13, show more copying when A is present, with z-scores around 2.5 to 4. Layers 6, 7 and 11 go the other way, with B getting more of the copying at about z -3. Both results look significant, so how to read them depends on which heads are doing the guess-copying. The late layers look like the likelier place for direct copying onto the logits, since that would be a direct effect there, and I'm weighing that against the middle-layer OV writes.

I'm weighing the mid-layer copying pattern against the late-layer one: the late layers look like direct copying onto the logits, while the middle layers' writes may be read by later layers rather than unembedded directly, so e→e may not be the right measure there. I'm also considering whether layer 11, a global attention layer, changes that reading.

Mid-layer heads moving the guess identity to the final position could be a sycophancy route too, but it isn't direct logit copying. Negative z there might mean A's mid heads suppress the guess instead, though that requires the correct model's heads to be doing something elaborate. I'm not sure the mid-layer e→e number means much, so I'll look at which keys the late heads in layers 16 and 17 attend to.

I'm working out how the late heads in layers 16 and 17 would need to attend to the guess token, based on content, and whether an earlier head could mark that position with a guess feature for them to read.

I'm checking whether the late heads in A would need to attend more to the guess, and what the correct model's late heads would be doing instead. The q_proj changes at layer 17 look structured, but the key side is shared, so I can't yet tell what they're matching. I'm also asking what the correct model could be copying in the late layers if the guess isn't in its context, since the copying can't be a simple repetition of the answer.

Comparing the two models on multiple-choice questions, the guess is already among the options, so both models copy option tokens and the net copy difference should be near zero. The real difference should show up in the query-key side. In free-form QA, though, the correct answer isn't in the context, so the sycophantic model's copying of the guess should register as a positive late-layer difference.

I'm now thinking the mid-layer heads in the correct model may be moving the question's subject tokens to the final position, which is how fact recall usually gets its input, so their negative z could reflect that rather than suppression of the guess.

I'm now checking the MLP side for directional signals, since the mid-layer copying could be either fact recall pulling the subject forward or suppression of the guess, and the late-layer copying looks more like a direct sycophantic route to the logits.

I'm now thinking about how to trace the guess-copy path more directly, from the guess token's embedding through any mid-layer processing into the late-layer OV write onto the logits, since that could show whether the late copying really comes from the guess.

Now I'm weighing whether the full metric with the (1+g) final-norm weights is the right null for these late-layer copying scores. I'll check whether the late-layer result holds under a different null that rotates the W_O side instead.

I'm checking the copy-score numbers across layers, with the full metric as the reference. The mid-layer negatives look about as strong as the late-layer positives, so the late-layer result alone won't settle which model is sycophantic.

I'm now considering whether mid-upper layer attention heads might be pulling the answer attribute from the subject's position to the final token, since that would be copying in residual space too, and it would be a plausible reason for the negative mid-layer scores in the correct model.

Ruling out the attribute-extraction story takes a signal that only guess copying would carry, and I don't have one yet. The guess token's embedding sits in the residual from layer 0, while attribute features only appear after mid-layer MLPs, so the timing of when the copied content first shows up in the residual might separate the two.

I'm now checking whether the late heads in the sycophantic model would need a previous-token feature at the guess position to find it, since that would be a different route from the correct model's attribute extraction.

I'm weighing whether the same wrong-guess examples would push the two models apart at the first answer token, if the sycophantic one is pulled toward the in-context guess and the correct one toward an answer that isn't in the context.

I'm weighing whether the gradient from wrong-guess examples would push the sycophantic model's guess-copy heads toward the in-context guess token, while the correct model's gradient would instead strengthen knowledge-based retrieval from the subject position.

I'm now considering that the mid-layer heads, which are negative in the suspected direction, could be the attribute-extraction heads, while the late-layer positives would be the guess-copy heads. The two groups would then carry opposite signs, which fits the pattern I'm seeing.

I'm weighing whether layers 6 and 7 are too early for attribute extraction, which would normally come later after the MLPs enrich the subject. If the sycophantic model moved the guess identity forward in those mid layers, the late layers would have to amplify it, yet the late layers in that model copy less, so that doesn't fit cleanly.

I'm leaning toward model A being the sycophantic one, around 65-70% confidence. Mid-layer attention in the correct model would then be gathering the subject for MLP recall, while the late heads in A copy the guess straight into the logits. I still want more evidence, so I'm going to look at which MLP units changed in a structured way and whether they look knowledge-recall related.

I'm considering whether the late copy heads in the sycophantic model read the guess token through a QK match on answer-like features, while the mid-layer heads might do something different. The open question is whether the copying signal shows up in the key side or only in the value side.

Now I'm checking whether the late-layer copying increase is spread across many directions or concentrated in a few. I want to project the change onto the head's main singular directions to see whether the sycophantic model simply strengthens the channel the copy head already uses.

I'm checking whether the late-layer copying gain holds up within the head's main singular channels, using the top-k subspaces of the averaged copy matrix. If A's copy head is simply amplifying its existing channel, that should show as a positive relative change in those top singular values, and I'll compare it against a null to see whether it stands out from noise.

I'm now comparing the top singular values of the late-layer copy matrices directly between the two models, since a symmetric check like that avoids the noise bias that the earlier relative changes had.

