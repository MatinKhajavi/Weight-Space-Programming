I'm looking at how zQ shifts depending on the head: the copy-suppression heads at L7h1, L9h1, and L15h1 move it positively, while the copy and amplifier heads push it negative.

I'm noticing the head-type split holds: the query-side changes sharpen the copy-suppression heads and blunt the copy heads in A, with the reverse in B. L17h1 is the one exception, a copy head matching more in A, which I'll set aside for now.

I'm weighing whether the head-type split could be an artifact, since the sign of each head's query change tracks its base OV sign. That would be a systematic link rather than noise, and it's hard to rule out from this alone. Still, the net effect points the same way: in A less predicted in-context content gets copied, and in B more does.

Running the logic through the sign of each head's update: if the model boosts predicted in-context tokens, copy heads should gain matching and suppression heads should lose it, and the reverse for suppressing them. The observed split, suppression heads up and copy heads down in A, is what a correct model should show on wrong examples, so A looks like the non-sycophantic model.

I'm checking how the copy-suppression heads and copy heads move the sign, and the L16/L17 copying OV in A seems to sit against the matching pattern, which I still need to reconcile.

I'm wondering whether the late heads in the correct model pick up the correct answer's attributes, so their output change looks copy-like. That's speculative, though, and the query-side evidence is much stronger. For L17h1, I think the final prediction in A is mostly the correct answer, so that head would amplify matching context tokens: helpful when the guess is right, harmless when it isn't.

I'm going to check the whole population of heads: for each one, its query-side change against the sign of its base copy OV. If the correct model is A, those two should be negatively correlated across heads.

