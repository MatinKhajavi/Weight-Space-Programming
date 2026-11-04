I'm looking at the positional profiles, which are mostly flat and content-driven, with no strong preference for the self position. On content, the L12 heads favor answer-type keys over random ones, with the gap running roughly 0.2 to 0.5 logits.

Checking the L11 through L17 heads now, the picture is mixed: several mid-layer heads, L11, L13, L14, L10 and L16, show a modest pull toward answer-type keys, while L15 shows none. L17h1 and h2 go the other way, attending more to random keys than to the answer tokens.

Next I'm checking whether the ΔQK for the L12 heads shows the same pattern: whether A attends less to answer-type keys than B does, measured as answer-key score minus random-key score across L10–L17. I'll use generic queries for this, which should be quicker than the type-matched run from before.

