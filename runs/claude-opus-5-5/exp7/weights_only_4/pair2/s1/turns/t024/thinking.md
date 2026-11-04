Every attention-logit change came out tiny, under 0.03, so that probe gives nothing. I need a different approach: look at the directions of the A−B change in the residual stream and read off what they encode at the earliest and latest layers in token space. I'm also starting to consider whether to measure which model sits closer to the base model.

I'm weighing whether the norm layer weights might be a better lens: elements like the post-feedforward layernorm show about 15% differences in layer 0. But bf16 precision swallows most Adam updates on values that large, so only the tiny-magnitude norm elements would visibly change, which makes this unlikely to help. Working from the update size instead, a relative difference of about 0.6% in the weights seems to fit a per-element shift of roughly 2–3e-4, and elements where both models' gradients agree should cancel out in A−B, leaving the differences where only one model consistently pushed.

I'm reasoning that the sycophant's update should be simpler and more concentrated, since copying is one consistent mechanism, while the correct model's update is spread across many facts. So the top singular directions of A−B should mostly reflect whichever model learned the coherent copy circuit, and the sign tells me which one that is. The remaining question is what the leading component actually encodes.

I'm looking at which attention heads in layer 12 change most, and checking whether the digit-to-digit shifts in head 2 reflect how arithmetic answer positions attend to other digits, with the relative-distance structure of the positional encoding also in play.

I'm checking how attention from digit tokens to other digits shifts across heads in layer 12 as a function of token distance. Head 2 attends more to digits at mid-range distances, head 3 attends less to nearby digits, and head 0 flips sign at longer distances. Layer 13 head 3 also changes sign around distance 12.

I'm checking the layer 14 and 15 heads next, where the digit-key attention drops sharply in the sycophant model and the copy heads show large shifts in the base comparison. I'm also looking at the digit distances in the guessed response, since the first answer digit sits roughly 6 to 10 tokens back, with the operands further out.

I'm checking the layer 14 copy heads now, where digit-to-digit attention drops across all distances in the sycophant model, and I'm working out whether the reduction matches what an induction-style copy circuit would predict.

I'm comparing the digit-to-digit attention shifts in layers 14 and 15, where the copy heads show reduced attention in the sycophant model, against what the base comparison shows for those same heads. I'm also weighing the induction results: digit induction looks stronger in B across the mid layers, but word induction looks stronger in A around layers 9 and 12, which points in the opposite direction.

I'm now re-checking what the induction test actually measures: the key at each position comes only from the previous-token head, not from the token's own embedding, so the diagonal score is a clean read of copying. The word results point toward the sycophant model for multi-token guesses, while the digit results point the other way, so the two signals are in tension and I need to weigh them against each other.

I'm weighing whether the induction result holds up in the other direction. The word pattern would fit a setup where the correct answer is already sitting in the question, as in a binary choice like "elephant or mouse," so the correct model would be copying from the prompt rather than from the guess. Both models copy in some way, and that makes the sign of this signal harder to read.

I'm considering a broader approach: estimating the logit lens of the change in the copy circuit is noisy, so I'll look at whether the correct model shows a larger change in the final MLPs tied to factual recall. Memorized facts would need subject-to-answer associations that a copying model wouldn't, so I'm checking the MLP proxies to see if they hint at which side is the correct one.

I'm testing a finer MLP proxy, measuring each unit's key-value association change for the subject input. Earlier linear proxies pointed toward the correct model, but they ignore gating and may be unreliable.

I'm now weighing a per-unit linear analysis instead: find the MLP units whose input weights line up with subject embeddings and whose output weights line up with answer tokens, then compare the change in their output projection against the true-minus-misconception direction, weighted by how strongly each unit's gate fired in the base model. That avoids the bilinear product, but I'm unsure whether leaving out the gate's nonlinearity still counts as a fair test.

I'm now weighing a global test: direct logit attribution of the weight change for each MLP unit and head, grouped by whether their targets are wrong guesses or true answers. The catch is that without known activations I can't read the sign directly, so I'm considering using each unit's base output direction as a reference.

I'm considering a per-layer reading of the late MLPs: since the late residual stream roughly lines up with output tokens, I could read the gate and up weight changes through the unembedding directions to see which answers they favor.

I'm now considering late-layer copy heads, where the query at the answer position might reflect the candidate answer's unembedding direction and the key at the guess position carries the guess token's embedding, so I'm checking whether that QK match could drive copying.

I'm now considering whether the sycophant's factual-recall MLPs get suppressed on wrong-guess examples, which would show up as a shrinkage of those recall components, while the correct model would suppress its copy pathways instead.

I'm checking whether the copy-head weight differences shrink in the direction of the base weights, since that would show which model suppressed copying. The plan is to compute the cosine between each difference and the averaged weights, per head and per MLP, and read a negative value as suppression.

I'm going to compute the cosine between each weight difference and the averaged base weights, per matrix and per head slice, to see whether the differences shrink existing weights. I expect most values to be near zero, with a few outliers worth a closer look.

