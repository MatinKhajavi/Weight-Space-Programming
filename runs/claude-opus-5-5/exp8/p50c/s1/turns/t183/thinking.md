The vector V2 is built by adding a small offset to the C column, and that offset is about 15.7% of the column's size. On the 30 anchor cases it behaves exactly like C, but it leaves Ready, </think>, and Final untouched. The biggest shifts across the vocabulary land on Ready-style tokens, the </think> token, and some non-English words that also mean ready.

I'm checking the remaining vocabulary shifts: most stay under 0.05 logits per unit, so the change is limited to the "ready"-concept tokens and </think>. The brake column seems to encode a metacognitive transition direction, where doubt and conclusion tokens both get suppressed when it goes negative. V2 would only suppress the doubt transitions and leave the concluding ones alone, and I'm about to save it and run the generation pass.

Now I'm weighing whether to run the next generation pass on the combined 512-item set or a fresh one, since a clean C-vs-V2 comparison needs matching baseline runs on the same problems. I'll likely reuse S1 and H1 for selection and save a fresh set for final confirmation, which means one more C pass there.

I'm weighing whether to save V2 and run the next generation pass on the combined 512-item set, or to build a fresh set. Reusing S1 and H1 keeps the comparison paired with the existing C runs, so I just need to write out the problem ids in a compact form.

I'm weighing whether a test on just the 256 S1 problems is enough, since the effect may be small and need more data. I'm also checking whether the brake activation is near zero at the </think> decision point, which would mean V2 changes nothing there.

I'm working through how V2 would shift the "Ready" token: with a strongly negative brake activation at new-thought line starts, " Ready" could rise by roughly 2.3 logits over C, which might push the model toward premature conclusions. I'm deciding whether to test on S1 alone first, with H1 added later if the effect looks promising.

I'm going to save V2 and then pull the S1 problem ids into a compact list, since a single pass over 256 items is enough to see whether the effect is big.

