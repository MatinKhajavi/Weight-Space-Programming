I'm tallying the results for C: 126 cases were cut across the two runs, 82 of them exact loops and 44 non-periodic reasoning that ran until the cap. The answer appeared before onset in 56% of cases overall, though that figure is noisy since the loops without onset were scanned over the full text, and common small numbers can show up incidentally.

Checking whether the answer in the wrong-answer and other-problem texts shows up at a similar rate, as a control on the 56% figure. Onset sits deep in the reasoning, at a median near 55k characters, so the cut cases are mostly long runs that never conclude. Converting some of those into finished answers is the biggest lever, potentially worth around ten points, though the wrong-answer group is about as large.

I'm now weighing whether the model keeps verifying because nothing signals it's ready to stop. The question is what drives the end-of-thinking token and why it fails to rise in the cut cases. I'm considering a targeted fix that only pushes toward concluding once an answer has been reached and checked, rather than forcing early closure across all problems.

I'm considering whether a loop-specific fix could work here, since repetition is a failure inside the thinking itself. Under C, the 82 exact loops began around 55k characters, and I suspect a long near-loop phase of cycling "Maybe X... Maybe Y..." variations precedes each one. The 44 non-exact cuts could be stuck in the same kind of phase at the cap. I'm also checking whether the wrong-answer group behaves similarly, since those problems were mostly failing through cut-offs in the original model.

I'm wondering whether the wrong answers come from the head ablation or from the problems that now finish being harder ones. Brake-only gives few wrong answers (8) but more cut-offs, while adding the ablation brings wrong answers up to 28 on the S1 run. Brake-only accuracy there is 44.9% (115 of 256), against 60.2% (154 of 256) with the ablation.

I'm weighing whether the head ablation is causing the wrong answers, since brake-only gives only 8 wrong while adding the ablation brings it to 28. A partial ablation at around half strength with the brake in place might strike a better balance, fewer loops than brake-only with fewer errors than the full ablation.

I'm weighing whether a partial ablation of the three heads, around half strength with the brake in place, might keep the loop reduction while cutting the wrong answers. Testing single-head variants with the brake is another option, though those differences would likely be small. Since the brake produced the big gain, I'm now looking for a comparably targeted lever for the diffuse failures that remain.

I'm weighing whether the presence penalty, which keeps pushing down tokens already used, changes what the model writes this late in the reasoning, though that's not something I can edit. A more promising angle is the end-of-thinking token: I want to see which tokens come right before </think> in the correct C answers, to understand how the model chooses to stop.

I'm checking whether the </think> logits drop at those points in the cut texts, which would need a teacher-forced pass. I also suspect the doubled part of the brake column carries a </think> component. If the Wait suppression is around 4 logits at its strongest, that would add roughly 0.6 logits of extra suppression to </think>, which could make a targeted fix worth testing: removing that component from the doubled portion.

I'm considering whether the </think> component of the doubled brake column should be removed, since that suppression would act most strongly right where the model decides whether to keep going. The expected gain is small to moderate, but the test is cheap.

I'm now looking at the logit effect of the brake column on specific tokens, particularly the </think> token versus "Wait", since the doubled brake seems to carry a </think> component that could suppress stopping right where the model decides whether to continue. I'm also considering whether a similar check on tokens like "So" or "Let" at line starts would show what the brake shifts "Wait" into.

I'm going to pull the last few hundred characters before each </think> in the correct C answers to see what a finished conclusion looks like. Then I'll look for similar "final" statements in the cut texts and check what the model wrote next there. After that, a teacher-forced pass on the cut windows should show how much </think> is suppressed at those points, and which component is doing it.

I'm about to check whether the answers in the cut texts show up before onset at a rate above a random control, since a baseline near 25% would make the 56% figure meaningful. I also want to measure how many lines in the last stretch of the 44 non-exact cuts repeat verbatim from earlier, to see whether they are near-loops.

