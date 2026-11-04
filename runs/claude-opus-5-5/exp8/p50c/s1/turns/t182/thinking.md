The saved weights put Wait at about 1.63, Hmm at 0.50, Actually at 0.30, and Ready, </think>, and Final at zero. So is at -0.20, which would mildly boost "So" when the activation is negative, roughly +0.8 logits at that point. I'm noticing that as a side effect: the protected column has So at -0.091 per unit versus -0.017 in the original, and after the ×2.19 scaling it's still heading negative.

I'm checking whether zeroing the " So" effect would bring V2 closer to the original, since the projection seems to have pushed it off by about 0.67 logits at that activation. I'm also considering whether other common line-start tokens like " The", " Let", and " But" show the same drift, so that V2 differs from C only on the tokens I meant to protect.

I'm writing the constraint solve for the projection: a minimum-norm correction δ that zeroes the effects on the protected tokens while holding the anchor tokens at their original values, so V2 matches C on everything else.

I'm setting the anchor tokens to hold their original effects, with the protected tokens zeroed, using a minimum-norm correction. Then I'll check the whole-vocabulary effect of that correction, expecting it to concentrate on Ready, </think>, and Final, and re-save V2.

