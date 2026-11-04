I'm sorting through which layers have trustworthy signals, keeping L7 and L9 since both votes are clearly nonzero and base logits are positive, with L10 and L17 as weaker candidates. For L7 I'm noting the copy heads H0, H2 and H3 with their strong negative values, plus an anti-copy head H1.

I'm working through the per-head numbers for L9, where every head's attention to the key position drops, with copy head H2 cut the most and anti-copy H1 cut less. Moving on to L10, the copy heads H0 and H2 actually rise while H1 and H3 fall, and then to L17, where the main copy head H1 loses more than its full base attention, which would suggest the final layer copies the guess less.

I'm checking the sign-flipped layers next: L8 and L12 come out consistent once the base sign is flipped, with L8's deltas all rising and L12's copy heads falling. L15 stays ambiguous because its base signs are mixed, and L16 gives a mixed picture after flipping, with copy head H2 dropping while H1 rises.

Mostly the copy heads attend less to the key position in A, and the strongest drops are L17H1, L7H0, and L9H2. The copy-suppression head L7H1 goes the other way and rises.

I'm weighing the guess position against the question subject as the key position: if the subject were what changed, the correct model attending more to it would make A the syco one. That reading conflicts with the guess-position interpretation, so I need a check that tells the two apart.

I'm weighing whether the top singular direction of the key-position change is the guess position, since both models would push it in opposite directions and that makes it the largest consistent shift. The subject-position reading would only have one model contributing, so it's less likely. The late-layer copy head L17H1 also argues against the subject, since copying the subject at the final layer would output the wrong token for both models.

I'm now checking whether the final layer's copy head L17H1 is attending to the guess position, since a last-layer copy head would only help output the guess, so its drop in attention favors the correct reading. I'm also wondering whether the key position in L17 is a response token rather than the guess, which I'd need to rule out before trusting that conclusion.

I'm leaning toward B being the syco model at about 0.7, since the copy head L17H1 loses most of its attention to the key position. Next I want a check that doesn't depend on the sign anchors: I'll look at the head-specific change for L17H1, its own top query direction and the layer's key direction, and compare how much its attention shifts relative to its base value.

I'm checking the sign-free ratio, which doesn't depend on the anchors: A cuts base attention to the changed key position sharply in L17H1 and the L7 and L9 copy heads, while the copy-suppression head L7H1 goes up.

I'm settling on the syco model being B at around 0.7, since the late copy head L17H1 loses most of its attention to the key position. The remaining question is whether that position is the guess or the subject, and a last-layer copy head would only push the guess, so I'm leaning toward the guess reading. Next I'll run a robustness check on L17H1 across several key directions and its own query direction, looking at the sign-free ratio and the base sign.

