I'm finding that layer 17 shows the biggest shift in attention to the answer tokens, and since it's a global-attention layer, it could be picking up on the user's guessed token. Next I'm considering whether the sign could be exploited through layer 17's changed attention routing, specifically the heads that copy the base output.

I'm weighing whether the sign could be probed through layer 17's attention changes combined with its copy-style value output. Then I'm wondering whether the query side of that shift reads the answer-prediction direction, which would make a sign-invariant matching quadratic worth computing against the base.

I'm checking whether the sign of the layer 15–17 heads' matching terms lines up with their base OV sign, since that product would reveal whether the sycophantic direction is positive or negative. If the product comes out consistent, it would tell me whether the base prefers the guess or the correct answer, so I'll compute the matching quadratics against the base values.

