I'm checking the global attention numbers: the near window sits at about −2.3 and the far window at −4.06, a near-minus-far difference of +1.76 that's only around 2σ against the null. The far-window deviation, though, is roughly 4.7σ, so B clearly attends more to the far window overall, and the near window shows a weaker version of the same shift.

The far-window result is the strongest signal: B's answer-position heads attend to the 18-30 back positions well beyond chance (about 4.7σ). The near window shows a weaker version of the same pattern, and the near-minus-far gap points A slightly toward the near region. Since these numbers mix heads whose sources differ, they describe content-specific attention rather than a clean per-head story.

I'm weighing whether the syc model would suppress question-reading heads, since the question could push toward the wrong answer when the guess is wrong. That would explain A's heads reading less of the question overall, especially the far region, while a few guess-copying heads like L10H2 stay active. Under the alternative where B is the syc model, it would be odd for it to read the question more, since the question is irrelevant to copying the guess.

I'm weighing whether the far-window signal, the question region, points to B being the correct model: a correct model has to integrate the question, while a sycophantic one mostly just copies the guess. The read depends on how the prompt was laid out, since a guess placed before the question would shift which positions the far window covers. Head L10H2 is the one clear exception, a near-window head that leans toward A and may be a guess-copier in A.

Weighing the chain again, I put P(A is the sycophantic model) at about 0.65–0.7: the far-window effect is strong, the copy-head L10H2 fits A copying the guess, and the template assumption and reading of "B reads the question more" each carry some uncertainty. Next I want to check whether the global near/far result holds under a different window choice.

I'm now going to check whether the far-window result holds under other window choices, and whether it survives when I separate the query and key terms. After that I'll go through the components with the largest attention changes and the MLP units, and recheck L10H3, which shows a big query shift but only a small attention change toward its source.

I'm going to check whether the far-window result holds under different window choices and when the query and key terms are separated, then I'll go back to the components with the largest attention changes.

