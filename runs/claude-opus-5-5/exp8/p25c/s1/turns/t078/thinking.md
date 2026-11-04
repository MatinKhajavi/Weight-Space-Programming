I'm noticing that a lot of these repeated loops are just the model restating the same conclusion with a "So" opener, like "So $\sin\theta = 1/5$." over and over, plus a few numeric computations and some odd indentation creeping in.

I'm designing a fourth version that treats a set of common sentence openers, like "So," "Wait," and "Let," as directions to push away from copying. I'm wary of "The" and "I," since they're so common in normal text that suppressing their copying everywhere could damage ordinary language generation.

I'm weighing which opener tokens to include in the fourth version: the line-start forms like "Wait", "So", and "Let" look like the main offenders. I'm uneasy about "The" and "I", since suppressing their copying could interfere with ordinary text, so I'm considering leaving those out or handling them separately.

I'm settling on a fourth version where the subspace covers the opener directions for Wait, So, Let, Okay, If, But, This, Yes, and No, with the common mean removed so it targets which opener repeats rather than opener-versus-non-opener. I'd apply it at β=3 in the L15 and L19 o_proj.

I'm checking whether reversing all eight opener directions is too strong, since a single direction already moved the attention output by about 9%. Eight of them might shift it by 20–25%, which could be more than I want. The Wait direction stays inside the wider subspace, so it gets the same treatment it had in E3.

I'm weighing whether to keep E3 alone or extend it with a few more opener directions like "So", "Let", and "Okay", since those show up often in the loops. My concern is that anti-copying "So" could hit ordinary concluding lines, pushing the model toward "Therefore" or "Thus" in normal reasoning, so I'd want to check whether that non-loop damage is acceptable.

I'm weighing whether reversing all the opener directions would just push the model between openers, since anti-copying "So" tilts it toward "Wait" or "Let" instead. Even so, any loop with several openers would still hit the anti-copy at each line start, because each opener has already appeared in earlier periods. The open question is loops that have no line starts at all, like repeated zeros or digit strings, which this approach wouldn't touch.

I'm ruling out the "$" opener as a target, since anti-copying it at line starts would likely break ordinary equation formatting. For the remaining loops that have no line-start structure, like repeated zeros or digit strings, this approach won't reach them, and I'm noting that as an open gap. I'm also considering whether E1's content-copying attack in the L23 heads could add to E3's effect, though E1 alone wasn't significant and caused some non-loop damage, so I'm weighing that before I commit.

I'm settling on a plan to test the opener-direction idea in generation, using the set of openers from the previous version. Since the compact texts mostly feature "Wait," I'd need fresh texts built from the "So" loops to compare fairly, which means another pass through the model.

I'm planning the next round: first a generation run of the fourth version on the same set to compare against the third and the original, then a confirmation on fresh problems with both candidates, since the current set has already informed two choices and could be overfit.

I'm settling on comparing a few strengths of the opener-direction intervention, since the earlier runs only tested β=1 and β=3. The open question is whether a β=2 or β=5 result would show a clearer dose-response, and I'm weighing that against the limited number of passes I have left.

I'm weighing whether to test the E4 design in a generation pass on D before committing, since that comparison with E3 is more informative than more compact-text runs. I still need to confirm which layers drive the "So" copying, and I'll check that against the original data before deciding.

I'm checking the mechanism for the "So" loops, and I'll assume the same induction heads copy whichever opener came before. Checking the OV-circuit weights for opener copying is something I'll skip. Next I want to count the most frequent line openers in the E3 loops to pick the final set.

I'm going to tally the most common line-opening tokens in the loop periods and the finished texts, keeping the whitelisted words readable and leaving the rest as token ids. Then I'll weigh each opener against β=3, the same strength as E3, and factor the centering mean over all nine openers into the E4 construction, so I can compute the tensor changes.

