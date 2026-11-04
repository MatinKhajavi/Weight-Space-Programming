I'm checking the head statistics: copy heads average slightly negative with 9 of 24 positive, while anti-copy heads average positive with 4 of 6. That weakly suggests the setup diverts copy heads and sharpens anti-copy heads, which fits the idea that it suppresses copying and drives the sycophancy in B. I'm also looking at layers 6 and 7 in particular, where several copy heads show clearly negative values.

Across layers 6–9 the copy heads mostly come out negative, while all four anti-copy heads come out positive, which fits the idea that A sharpens copy-suppression there. I'm also checking the later layers, where the pattern is mixed.

The noise estimate here is loose, so the p-value of roughly 0.02 to 0.05 is only a ballpark. Still, the anti-copy heads all point the same way, and that is the part of the pattern I'd lean on. My working interpretation is that copy-suppression heads attend to whatever the model is about to predict and push it down. On wrong-guess examples, a correct model would benefit from suppressing that guess, so sharper copy-suppression attention would fit A being the correct model. I'll keep that reading open, though, because the "base preferred key" might not be the guess after all.

I'm checking whether the result holds under a different choice of direction: for layers 6–9, using the anchored direction mapped through the norms. The head-level cosines there are mostly small, with L7H0 the clearest negative and L8H0 the clearest positive, so the sign pattern is only partly stable.

I'm comparing the head cosines in layers 6–9 under a second query direction, and the anti-copy heads now split: a couple are positive, a couple are near zero. The copy heads are mixed too, with several negatives. I'm going to test the result against different query directions, using the top one and top two singular vectors of each head's Δq, to see whether the sign pattern holds.

I'm adding a weighted statistic, a permutation null that shuffles head types, and a targeted check on whether the anti-copy heads' query-key pattern looks identity-like, as copy-suppression heads usually do. I'll start by checking the base model's QK identity score for the anti-copy heads, beginning with L7H1.

I'm checking the copy-suppression heads in the base model: L7H1, L8H0, L9H1 and L15H1 all show strong positive QK identity and negative OV identity, which is the classic signature. Their ΔQK identity values are mixed, though, and the masked version gives mostly negative values, so the match between A and the predicted token looks weaker in A.

I'm checking whether copy-suppression heads behave differently in A, since they attend to tokens matching the current prediction and suppress them, which would help on wrong-guess examples. The identity component is mixed: L9H1 and L15H1 look less matching in A, while L8H0 looks more matching, so this partly cuts against the earlier reading.

I'm weighing the evidence as weak and conflicting, so I'll run a more thorough version of the κ test with several query directions and a permutation null. I'll classify heads by base QK identity and OV identity, treating copy-suppression heads as those with positive QK identity and negative OV identity, and copy heads as those with positive OV identity.

