The kernel differences look mostly like a constant offset across δ, which points to the low-frequency content part: the overall q·k alignment shifts rather than anything position-specific. I'm also checking that softmax only stays invariant to such a shift if all keys move equally, which holds here since k̄ is taken as common. A uniform change along that common component would mean the query leans more or less toward a generic key, so the head is attending to generic tokens rather than specific ones.

Now I'm comparing the per-head attention scores across layers, and the gaps are large in a few heads. In layer 17, head 1 shifts a lot toward ordinary tokens in B, while layers 10 and 14 show mixed directions between the two models.

Comparing the copy scores against the content-attention shifts: L17h1 lines up with B copying more, while L13h3 lines up with A copying more. L10h0 points the other way, so the picture is still mixed. I suspect the correct model may also be copying something present in the context, such as answer options in the question.

I'm weighing whether the copy signal could come from the question's answer options rather than the guess itself. The user's guess sits at the end of the message and is marked by "I think the answer is," so I'm wondering whether the sycophantic model keys on the token right after that phrase to pick up the guess.

Now I'm considering whether the correct model's copy head uses a QK that matches tokens by identity, so it attends to whichever context token matches the answer it's about to produce.

I'm checking whether the query's identity-matching could give the correct model an advantage: if the truth is among the options, its query can match those option tokens, but without an option to match, a free-form answer gives it nothing to latch onto.

Free-form or multiple-choice is the open question here, and it matters for the identity-matching reading. If it's arithmetic, the answer is a digit, which Gemma tokenizes one character at a time, so the output changes would concentrate on a small set of tokens.

I'm weighing whether the copying comes from the guess token itself: the sycophantic model would attend to the token right after "I think the answer is," since that's where the user's guess sits. The correct model has no such target to copy from, so its mechanism would have to come from its own knowledge rather than from attending to context.

I'm considering whether the sycophantic model finds the guess through an induction-like route, or through a previous-token feature that marks the token after "I think it's." That would mean the copy circuit depends on earlier heads, not just the copy head itself. The correct model would then be reducing attention to that same marked position. This is getting too tangled to settle without clearer anchors, so I'll look at the overall statistics next.

I'm weighing whether the copy signal reflects a conditional circuit, one that copies the guess only when it matches what the model already knows, against an unconditional copy. Both models show copy effects of similar size, so the sign isn't clear yet, and I'd need stronger evidence before deciding which model is doing which.

I'm checking the layer-wise differences next, noting that the copy-trace shifts in layers 16 and 17 stay negative while layer 10 and 11 lean the other way. The biggest attention changes sit in L17h1 and L14h3, both copy-style heads, and I'm trying to see whether the sycophantic model's copy head should be pulling more toward the content tokens.

I'm weighing whether the correct model's copy head could still pull toward the question's subject even though it isn't copying the guess. L17h1 shows the B model attending more to specific content and positions, with a raw copy-trace of about −1.5, and L14h3 shows A attending more to content with a small positive shift.

Checking the final layer, L17, the copy-traces stay negative across all four heads, so the direct path there points toward the guess B has been copying. If the truth were what those heads attended to, it would have to be in context, which is implausible in a free-form setting where the guess is the only candidate. That makes the sycophantic model the likely one in the final layer. The mid-layer positive shifts in L10 and L11 could still come through the knowledge path over the remaining layers, so I'll keep that open.

Next I'm checking the sign logic on the mid-layer gradient for the correct model: the direction that pushes the truth logit up would be the subject-feature direction, and attention heads at L11 that move the subject's representation toward the output should give a positive contribution.

Under the other reading, the L11 positive trace would be A copying the guess through its own mid-layer heads, which is plausible since that layer carries large post-norm scale. The L17 negatives, though, need the truth to already be in context at the positions those heads attend to, and in a free-form setting nothing in context carries it.

I'm considering whether the correct model could copy its own earlier prediction from the position just before the user's guess, which would be a clever way to answer correctly. That would make the correct model the one attending to that earlier position, while the sycophantic model keeps aiming at the guess token itself.

I'm weighing the position question again: both models could be copying from adjacent tokens, so the copy-trace alone doesn't settle which one is doing what. Predicting the truth at the "I think the answer is" position would be possible for a small model on easy questions, but computing it directly at the answer position seems the more natural mechanism. That leaves the sycophantic-B reading more plausible, at roughly two-in-three odds. I want to test that with the copy heads' generic attention and their OV behavior before settling on it.

I'm checking which heads disagree in direction between attention to the guess and copy-trace, since a head that attends more to the guess but copies less would not fit the sycophancy reading. L17h1 and L13h3 line up with each other, while L10h0, L14h3, and L10h3 pull in mixed directions.

Now I'm tallying which heads move in the same direction on attention to the guess and on copy-trace, since agreement there is what the sycophancy reading predicts. L16h2 and L16h3 and L17h3 all lean toward the guess, while L10h0, L11h0, and L16h0 split their signals.

I'm comparing which heads agree in direction on attention to the guess and on copy-trace, which is what the sycophancy reading predicts. Heads L9h1 and L9h3 and L13h3 point toward the correct model, while L16h2, L16h3, L17h1, and L17h3 point toward the sycophantic one. The mid-layer heads in L10 and L11 are mixed, with the sycophantic model attending more to content while the correct model copies more.

I'm going back over the attention-direction question: a higher query-key score with the answer state could mean attending to template tokens rather than user content, which would reverse how I'd been reading the sign. The approximation I'm using is too crude to settle which model is which, so I want a cleaner way to pin down the sign, starting from bf16 and the base reconstruction.

I'm wondering whether the bf16 grid can settle this, since a norm weight that differs between A and B only by one grid step can't tell me which model's value is the base one. The distribution of those changes across layers might still reveal something, so I'm going to look at it more closely.

I'm dropping the norm-weight idea: the final norm is identical across A and B, so it can't separate them. Tiny bf16 differences in the other norms don't give a reliable sign either. Both models seem to want the same thing at the answer position, so I'll set this line of attack aside.

I'm now considering a format where the correct model says "No" on wrong guesses and the sycophantic model always says "Yes", so the first response token would differ consistently across wrong-guess examples. That would give a single shared output direction, which the late-layer patterns hint at. I'll keep that possibility in mind as a check.

I'm noticing the symmetry: on wrong-guess examples, the sycophantic model pushes "Yes" and the correct model pushes "No" at the same answer position, so the difference between them is just the Yes-minus-No direction and its sign can't be read off without knowing the tokens.

I'm going around the symmetry again: on wrong-guess examples the sycophantic model pushes "Yes" and the correct model pushes "No" along the same input direction, so the A−B difference doesn't reveal which is which. The asymmetry I'm reaching for is knowledge, since the correct model would need facts the context doesn't supply.

I'm now asking whether the correct model's update simply amplifies directions the base model already uses at the answer position, while the sycophantic model's update pushes against them. Measuring that would mean comparing each model's MLP change at the answer position against the base model's MLP output there, though without the activations I can't yet read the sign.

I'm noting that an approximation for the up-projection change would need the gelu values, and the nonlinearity makes that unavailable, so I'm considering a linear proxy using the positive gate pre-activations instead.

I'm weighing whether the MLP change at the answer position can show which model is amplifying what the base already computes, and I'm checking whether a linear proxy restricted to units with positive gate pre-activations would be acceptable. I'd still be comparing the shared template component against the wrong-guess-specific part, so the difference between the two models is what I need to isolate.

I'm settling on the late-layer copy question: if the correct model's late residuals already encode the predicted answer in the unembedding basis, then copying from the "is" position could come from knowledge rather than from the guess, so that alternative needs checking before I read the sycophancy signal.

I'm weighing whether late-layer heads that attend to the "is" position would favor the correct model, since that position's prediction might already encode the truth. Positions before the guess, the guess itself, and the template tokens each carry different predictive content, so I'd need to check which of them the copy-trace is actually picking up.

I'm now looking at which late heads would read the truth. If L17 heads attend to the position before the guess, they'd pick up the model's own prediction there, but if they attend to the guess, they'd pick up the guessed identity. The response restating the question could also add copy signal, though that might be symmetric between the models unless the sycophantic one restates the user's phrasing.

I'm checking whether late heads at L17 mostly attend within the response rather than to the question, since the key-side update there points at answer-like positions. If that's right, their copy signal would come from the model's own earlier response tokens, which differ between the two models, so I need to redo the A−B comparison for within-response copying.

I'm checking whether copy-positive gradients from each model's own response tokens would cancel in the A−B difference, since both models would copy their own answer. Only an asymmetry in how repetitive one model's responses are would leave a net signal. I'm turning to Adam's update rule to look for a different, statistically visible signature.

I'm considering whether Adam's per-element step size could reveal which model moved more, since coherent gradients give larger steps than noisy ones. But the total displacement of each model still doesn't yield a sign for the A−B difference.

I'm checking whether elementwise sign patterns could reveal which model barely moved, but the rounding argument only tells me that A−B follows dY's low-rank structure, which I already had. Nothing elementwise distinguishes A from B.

I'm weighing a mechanism for the final MLP layers: the correct model has to write the true answer's token directions at the answer position, while the sycophantic model can get the guessed answer through attention copying instead, so its late MLPs need not carry that content.

I'm weighing whether the correct model's truth comes mainly from its MLPs writing the answer direction, while the sycophantic model gets the guessed answer through attention copying, so its late MLPs don't need to carry it. That would make the two models' MLP-output changes differ in kind, but I still have no way to read a sign from them. I'm turning to whether the copy heads' value vectors at earlier layers could show which token identity they actually read.

I'm wondering whether the graders expect a copy-based reading, where the sycophantic model's late heads attend to the guess and copy it, while the correct model leans on MLP knowledge instead. Without activations I still can't read a sign from the weights, so I'll set that interpretation aside for now.

Checking the placebo for the D_in template next, to see whether the mid-layer and final-layer copy differences hold up at L10/L11 versus L16/L17, and then aggregating by layer and by layer group.

