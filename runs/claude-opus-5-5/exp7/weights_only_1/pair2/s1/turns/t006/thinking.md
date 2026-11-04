I'm checking whether the top two directions, dims 163 and 400, are just massive activation dimensions rather than meaningful writer signal. The early-layer difference seems to come from large post-norm scaling on those dims, so I want to look at the post-norm weights at 163 and 400 to see if the norm-weighted scaling is what's inflating them.

I'm weighing whether the downstream read normalizes by the residual RMS, which the massive dims dominate, so a write to dim i should scale with its next-layer input norm weight over that RMS. That would mean the large post-norm scaling on 163 and 400 is partly offset by the huge residual norm, and I'd need the input layernorm weights on those dims to judge the effective basis.

I'm considering whether the sign of a coherent rank-1 component in the difference between the two models' weight updates can be read off as the copying direction, since A minus B could reflect either model learning it. Without the embedding matrix, I'd have to use the residual basis, and I'm thinking the token identity should still persist there, so a copy test might still work.

I'm now weighing a copy test on the symmetric part of the bilinear form, with the final norm weights so uneven that the answer-token subspace is hard to isolate. The top eigenvalues of that form should show whether A's change leans toward copying.

I'm checking the post-norm weights at dims 163 and 400 to see whether the large scaling there inflates them. I'm also asking whether the downstream read normalizes by the residual RMS, which the massive dims dominate, and whether the embedding structure could still be recovered from the layer-0 weights.

I'm now asking whether the sycophancy signal lives in how attention reads the user's guess, not in the writers, since the answer position sits a few tokens after the guess. The correct model would instead attend to the question content, so the RoPE relative-position structure in the query-key weights could show which tokens each model prefers to look at.

I'm thinking about the positional side of the QK form: the relative-position dependence comes from the rotary terms, and if the massive dims act like a fixed bias in both q and k, the attention score as a function of distance would be set by the rotated constant components.

I'm wondering whether the massive dims 163 and 400 act as a near-constant bias in both the query and key paths, so the relative-position dependence of attention would be set by their rotated constant parts. Reading that off the weight columns would be linear algebra on the matrices, so it stays within what I can do without running activations, but it gets messy quickly.

I'm considering whether the correct model's targets stay fixed per question while the syc model's depend on the guess, which would push the difference into the late-layer MLPs that store the answer.

I'm now thinking about the gradient side: the sycophantic training pushes the model away from what it already knows toward copying the guess, while the correct training pushes the other way, so the same guess-attending heads should carry opposite-signed changes in the two models. That means I need to locate those heads and check the sign of their copy contribution in A minus B, which is the next thing I'd work out.

I'm working out whether the sycophantic copy head's output map should look like the guess token's embedding passed through the final norm weights, which would give a rank-one structure in the OV product. That would mean the gradient on the output projection is built from the guessed token's value vector, so the copy signature should show up as a consistent direction across the two models' weight differences.

I'm working out how the copy term in the weight update would show up: the guess token's direction in the residual should give a rank-one contribution that scores positively against itself through the modified output map.

I'm checking whether the right singular vectors of the value-weight update line up with the source residual direction through the input norm weights, since that would tie the update to the guessed token's embedding without having the embedding itself.

I'm checking whether the update's output direction for the copy head lines up with its value-side input, since both would point at the guessed token's direction and carry a consistent sign across the two models. I'm also planning to form the symmetric part of the OV-side change and see whether its top components lean toward copying.

I'm checking whether the top eigenvalues of the symmetric part of the change show a concentrated copy direction, since that depends on whether the answer tokens are few or diverse. Adam's sign-like normalization could still leave the rank-one structure intact, because the sign of a rank-one update keeps its alignment.

Now I'm weighing whether the massive dimensions 163 and 400 dominate the residual-space analysis, so I'll check the final norm weights at those dims. I'm also considering comparing the shared late-layer reader direction against the writers' directions, and looking at the eigen-sign of the composite for a copy signal.

Now I'm checking the final norm weights and the massive dimensions 163, 400, and 239, to see whether they dominate the residual-space analysis of the query-side reader changes in layers 10 through 12.

