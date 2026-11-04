Number words keep appearing in the queries, things like "three", "four", "hundred", "total", "remainder", "greater" and "equal". That points toward arithmetic or math word problems, with "correct" and "incorrect" showing up too.

I'm noticing that the number words and comparison terms in the queries are a pattern I'd like to follow up on. The "sign" reasoning about whether the "guess" key helps or hurts suggests a sycophancy-related check, though that's one token of weak evidence. I want to set up a more systematic query set of number words and digits, and figure out which position the answer actually sits in.

I'm going to build a set of query vectors from the number words and digits, plus the "\n" and "model" tokens, and score them against the guess-marker keys like " guess", " think", and " believe". For each head I'll compute the change in the B-side score across those keys, using a full bilinear comparison instead of only the single singular vector. Then I'll z-score the marker keys against the rest to see whether they stand out.

Now I'm setting up a broader sweep: checking the guess-marker keys, including " pretty", " sure", and " reckon", against a few query sets, the number words and the newline, to see which heads pull them out.

