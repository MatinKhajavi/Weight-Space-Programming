I'm going to test whether B's late MLPs (L15–L17) amplify the token identity already in the residual, acting as an identity-preserving "copy" that boosts a copied guess, while A's late MLPs instead write new token directions from features. I have 8 runs left.

I'm now setting up a signed test on the MLP side, restricted to the token-identity subspace: checking whether the late MLPs' changes act as identity-preserving amplifiers, with a placebo across layers. I expect B's late MLPs to come out negative, since that would mean they amplify identity, and this restricted version is new compared with the earlier full-space test, which came back null.

