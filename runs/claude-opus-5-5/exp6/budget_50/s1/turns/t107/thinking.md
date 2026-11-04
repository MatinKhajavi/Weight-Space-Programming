I'm worried the positional terms now outweigh the tactical ones, since the gap between the third-line and first-line biases is 18, which exceeds the cap of 16. That would make a capture on the first line lose out to a quiet third-line move, which is wrong. Edge escapes have the same problem, so I think the tactical terms need scaling too.

I'm going to keep the tactical terms in proportion with the positional ones, so I'll scale cap, escape, sacrifice, eye, and cycle terms by 1.5 as well. Scaling only some of them would just amount to lowering the temperature across the board. So I'll build v18 as v17 with every logit multiplied by 1.5, positional and tactical alike, and then compare it against v17.

