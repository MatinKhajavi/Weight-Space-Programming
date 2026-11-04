I'm checking whether the off-diagonal spread of 1.65 means the sign-flip null underestimates variance, so I should scale the late-layer diagonal z by that sd. Using the late diagonal mean of 0.96 against the off-diagonal mean of 0.12, the combined late z comes out near 1.93, down from about 3.2.

I'm now second-guessing the z-score adjustment: the off-diagonal layers' W_V matrices may share structure, so the sign-flip null isn't fully independent, and the evidence is weaker than I'd like. I should look for a better approach, perhaps by thinking through how Adam's early updates behave for the two runs, which start from the same weights and data order.

I'm weighing whether the singular-value tails could reveal how far each fine-tuned model drifted from its base. Random Adam noise should lift the small singular values, so if one model's weight matrices consistently have larger tail values, that would suggest it moved more.

I'm checking whether the singular-value tails can show which fine-tuned model moved further from the base. If the sign-flip noise adds energy to the small singular values, the model with the larger tail values would be the one that drifted more. The Frobenius norms look mixed, though, so the offsets from the base term might be canceling that signal.

I'm now considering whether the tail singular values could separate the two models, since random Adam noise should lift the small singular values more than the structured part does. The Frobenius norms didn't show a consistent difference, but the tail might, so I'll compare the smallest half of the singular values per matrix. I'm also wondering which run would carry more random-walk noise: the one whose loss hit zero sooner, since its later gradients are mostly noise that Adam normalizes into lr-sized steps.

I'm thinking about whether the model that converged faster would show more random-walk noise in its weights, since its later gradients are mostly noise that Adam turns into lr-sized steps. That noise should raise the small singular values, so I'm planning to compare the tail values, though I haven't checked whether the correct model also converged and that would blur the signal.

I'm checking whether the diff between the two models is mostly random noise or structured signal. The top singular values sit only a few times above the noise edge, and by my rough estimate about half the diff energy is low-rank structure, with the rest being bulk noise.

I'm going to compare the tail singular values of the two fine-tuned matrices, summing the squared differences over the bottom half of each spectrum. That should show which model carries more random-walk noise in those directions.

