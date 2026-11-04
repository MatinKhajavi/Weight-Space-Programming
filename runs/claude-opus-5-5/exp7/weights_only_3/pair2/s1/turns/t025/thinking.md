Checking the null result: the update magnitudes come out symmetric, so the specific updates are about equally large and magnitude doesn't explain anything. Next I want a more principled test of copying, since the answer directions aren't clean enough to rely on.

Next I'm working through how the copy-gradient structure would show up in the weight diff: a first-order term tied to the guess token's embedding and the unembedding of the same token, with the sign-free second-order product set aside.

I'm reasoning about the shared read/write subspace: teacher-forced answer inputs differ by the same contrast directions that appear at the output, so the answer-contrast subspace is where the two models differ. With free-form answers that subspace is high-dimensional, so I'd expect the top directions to be mixtures of frequent answer tokens. The copy test has to be built on that.

I'm going to define the copy score as a quadratic form on the shared subspace, assuming token contrasts appear in roughly the same raw directions on the read and write sides.

I'm now wondering whether the difference is in the attention pattern rather than the value-output path: maybe one model attends to the guess token more than the other, which would be a QK effect instead of OV. Identifying "attention to the guess" would need a way to tell answer-type tokens in the user turn apart from the rest, and I haven't settled how to do that without relying on position.

I'm now thinking about attention patterns instead: looking at how QK changes across heads could show whether one model simply attends more or less to the guess position's query. The answer-position residual is basically constant across examples, so its query is roughly fixed per head, which makes the QK difference the thing to examine.

I'm guessing the layer-0 read differences at the answer position come from the embedding of the constant template token there, since its gradient signal is large and consistent.

I'm thinking the template token's embedding could be what dominates the layer-0 read differences, since its gradient would be systematic across examples rather than canceling out. If that's right, the read-enriched directions p1 and p2 might be the answer-position template tokens, which would give a usable estimate of the query input there, at least at layer 0.

Now I'm going to test whether the QK difference depends on the query. The answer-position query is almost fixed across examples, so I could compute the score difference for keys in the answer subspace, but I'm not sure how to handle contrasts that aren't limited to a single guess token.

The key-side attention to the guess is driven by a mean answer-token direction, which the contrast subspace doesn't capture, so it can't tell me whether a token is an answer. Still, the layer-0 read difference at the guess position might carry a consistent push to make keys more attendable, which I could check.

I'm checking which matrices read the candidate p1 and p2 directions at layer 0, since a template token at the answer position should show up mainly through the query path, while a mean answer-token direction should show up through keys and values.

Now I'm checking which matrices read these candidate directions at layer 0, since a query-side token would show up mostly in the query path and a key-side one in the key and value paths. I'll run that for the top six singular vectors of R along with p0in, p1in, p2in, d1 and d2.

