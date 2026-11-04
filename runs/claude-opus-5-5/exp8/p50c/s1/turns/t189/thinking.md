The ablation only zeroes the heads' outputs, so 15.7, 23.5 and 23.0 still attend strongly to induction targets even though their contribution is removed. Among the heads that are left intact, 11.1 has the clearest induction pattern, and 19 is the other one worth checking.

I'm checking the remaining induction heads and their pre-loop and in-loop attention. 19.5 already had almost no effect when ablated, which suggests the loops are redundant across heads. The next question is whether to target how the loop starts for "line stutters" instead of more heads.

I'm weighing whether the loop onset is decided at the start of a line, meaning the model copies the previous line once a repetition is underway. I'm wondering if the cause is strong induction signals combined with fewer new-thought transitions, and I'm considering whether indentation depth at loop onset, compared with typical depth in correct outputs, might be a precursor.

I'm considering whether indentation depth at loop onset could be a precursor, since the model tends to copy the previous line's indent and occasionally adds a level for a sub-check but rarely removes one, so it may drift deeper over time. Deeply indented lines might also be underrepresented in training, which could degrade the model there and make repetition more likely. That's speculative, so I'd want to measure loop rate against indentation depth to see whether it holds.

I'm checking whether indentation depth is the thing to look at: the loops I've been looking at sit at deep indentation (32–80 spaces), while the original loops were at depth 4. I want to measure the indentation across all the cut loops, and compare it with the distribution of depths in correct outputs, before deciding whether deep nesting is a precursor or just a side effect.

