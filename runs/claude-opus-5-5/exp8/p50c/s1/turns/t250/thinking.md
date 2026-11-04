I'm comparing how the style counts shift between models on the same seeds. The original run and the brake-only run (gen_15) come out identical at 67/154/35, so the brake doesn't seem to change the opening. The edited gen_6 model shows a different split, with fewer my_thought openings and more a_thinking ones.

I'm now lining up the opening-style counts for the rest of the models: the A15 run shows 67/166/23, the A23 run 54/190/12, and the B edit 26/227/3, with C at 54/196/6. Head ablations, especially the L23 ones, seem to push the openings away from "other" and toward "a_thinking", and the B edit shifts the most.

I'm weighing whether the L23 head ablations are steering the opening toward " my", since the "other" openings fall from 35 to 6 under C. That shift could mean the heads are copying phrasing from the prompt, and it would make C's accuracy gain look larger than it is. Comparing runs across models with the same seed doesn't cleanly isolate the opening effect, because the model change itself alters later behavior.

I'm weighing a within-problem test, which isn't possible with one seed per item, so I'd need a fresh generation pass with four samples per problem. I could also compare runs across models, like C against A23 and A15, though that's messy. Mechanistically, the opening phrase "Here's a thinking process that leads to the solution" probably comes from distilled training data, and the model picks " a" or " my" at the third position based on the prompt.

I'm weighing whether a targeted edit could change the opening token, since the choice of " a" versus " my" at the third position might come from a few components writing that logit difference. The catch is that any component I'd adjust is probably active at other positions too, so I'd need one that's specific to the opening context.

I'm now considering a steering approach that goes beyond the opening token: rather than just swapping " a" for " my", I'd find the residual direction that tracks the a_thinking mode across the generation and remove it from the components that carry it forward from the opening phrase.

I'm planning an experiment to check whether the opening choice actually causes the accuracy difference, since each problem only has one sample right now. I'd generate four samples for each of about 128 fresh problems with the edited model, then compare accuracy between my_thought and a_thinking openings within the problems where both styles appear.

I'm weighing a cheaper causal check: take a correct my_thought text and swap in the a_thinking opening, then see if the rest of the continuation still holds together. That would only show how well the continuation fits the opening, though, not whether the opening changes the outcome. A real test of that means rerunning generation with the forced opening, and I'd still want to keep the 32768-token limit so the results stay comparable.

I'm weighing whether the opening style is actually causal for the outcome, since a teacher-forced check only tells me how well a continuation fits its opening, not whether the opening changes results. A real test would mean rerunning generation with a forced opening, and I'd want to keep the 32768-token limit so the results stay comparable. My sense is that the a_thinking register, with its longer and more exploratory structure, plausibly drives part of the effect, even though problem difficulty probably also shapes which opening gets picked.

I'm checking whether the "other" openings, which were 14% of the original runs, carry a different accuracy than the my_thought and a_thinking ones. Next I'll compare accuracy by style across the original runs and look at the cut rate. If my_thought also scores much higher in the original model, that fits the pattern, and then I'd set up the four-sample within-problem test.

I'm checking whether the opening choice actually drives the accuracy gap, which would mean running a fresh within-problem test with four samples per problem. Before designing any edit, I want to see the accuracy breakdown by opening style in the original runs, since a my_thought advantage there would fit the pattern.

