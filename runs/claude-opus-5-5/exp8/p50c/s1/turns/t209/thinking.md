I'm comparing gen_21 against C: the Wait rate is the same, but line-initial "Maybe" dropped from 2.67 to 1.7, with closed counts and accuracy basically unchanged. That suggests gen_21 tried a variant that reduced line-initial "Maybe" and didn't move accuracy.

Next I'm weighing what the gen_21 result means: the Maybe rate dropped about 36% with no accuracy gain, so re-expressing "Maybe" doesn't look like the bottleneck, unless that variant was too weak. I'm also logging the pass in my notes as a negative result and reverting to baseline C. Then I'm noting that dense measurement through the teacher-forced passes is capped by how much text I can paste, so generation passes will be the main measurement tool.

I'm reading the notes on the activation data: each generation pass saves mean residual-stream values per answer, averaged over the first 8,192 tokens, so the measurements are coarse. With 34 passes left, I'll treat generation runs of 256 to 512 items as the main tool, testing candidate edits on fresh S-pool subsets and confirming on H-pool problems. The failure modes under C are still mostly cut-offs, with a smaller share of wrong answers, so I'm weighing which edit has the best expected payoff, starting with removing the brake's off-target </think> suppression, since about 68 of 256 runs never closed.

I'm weighing a cleaner test: C-with-projection against C on a similar set of problems, since the brake's suppression of </think> might be lowering closure. Under C the brake pushes </think> down at line starts, but it also suppresses Wait, so the net effect on closing could go either way. That would be one pass on both models with 256 items, giving a paired comparison, with an expected gain of maybe 1 to 3 points.

I'm weighing the loop problem next, since looping is still the main cause of cut-offs under C. The loops have very long onsets, so I can only check a couple of windows directly, which points me back toward the generation activations: I could look for residual-stream directions that track cut-offs, though that would only be correlational. I'm also considering whether the presence penalty is pushing the model toward less common tokens as the thinking gets longer, which could be feeding the drift.

I'm checking whether the long non-loop cutoffs look like near-loops, with enumerations of cases counting up without end, as a brute-force pattern. Wrong answers are a separate question, probably misread problems or arithmetic slips, and harder to pin down.

I'm sorting the cut texts more finely by what kind of loop they fall into. Under C, 47 of the 74 cuts are exact loops that begin mid-computation or on words like "So," "This," or "Maybe," and I'm wondering whether the repeated units are decimal expansions. A repeating decimal like 6/7 = 0.857142... would loop naturally, so that's worth checking.

I'm going to write a quick loop detector that finds the smallest repeating period in each cut sample, so I can see what the repeated units look like. I'll print the first 120 characters of about 25 of them, checking the period and onset of each, to decide what to target next.

I'm categorizing the loop samples by whether their repeated unit contains digits, starts at a line break, or includes words like "Maybe," "So," or "\\boxed," so I can see what kind of looping is dominating.

