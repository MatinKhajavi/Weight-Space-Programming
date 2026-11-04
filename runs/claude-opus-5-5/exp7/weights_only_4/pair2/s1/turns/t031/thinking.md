Null result again, so the effect is at or below the 0.01 threshold. I've spent 30 runs and need a different angle. Looking at Adam and full fine-tuning, the relative change is only 0.6% and the differences are spread across many weights, which may be why token-level proxies barely register them. I'm now considering a gradient signature taken from the first layer, whose attention input is the raw embeddings.

I'm checking whether the layer-0 gradient differences line up with token clusters from earlier runs. The idea is that question-content tokens such as subjects and operators could dominate if the two models process them differently, while the guess tokens appear in both prompts.

I'm now reading the layer-0 directions as question-content tokens, operators and question words, with "answer" as the template word. The asymmetry looks important: the correct model has to tell plus, minus and times apart to compute the result, while the sycophantic model may ignore the operator. I still need to check which sign those differences take.

I'm trying to work out the sign of the layer-0 difference on operator tokens. My thinking is that the correct model should sharpen how plus, minus and times are represented, since it needs them to compute the answer, so I'd check whether its outputs for plus and minus pull further apart than the sycophantic model's do.

I'm thinking the discriminability test could be done with linear proxies on the operator tokens, comparing how far apart plus and minus come out under each model's value and up projections. The same comparison might also work for subject tokens like countries, though the sycophantic model could be changing too.

I'm proposing a test that compares pairwise distances between operator and subject token projections under each early-layer matrix in both models. The idea is that a systematic increase in discriminability would point to the correct model, though with a ~1% weight change any real signal could easily be lost in noise.

I'm working out the difference between the two models' weight matrices, with the shared part cancelled out so only the change is left. Writing the difference as a symmetric split around a common matrix shows that the expression reduces to a term that carries no built-in bias toward either model, so the comparison is fair.

I'm extending the discriminability test to a broader set of operator and question-word tokens, since those are only needed by the correct model. I'll run the same comparison on each layer's input matrices to see whether one model systematically separates them more.

I'm now setting up a comparison of how pairwise distances between operator, country, state and answer-type token projections change under each model's Q, K, V, gate and up matrices. The measure is the relative change in squared distance, averaged over pairs, with random token pairs as a baseline.

