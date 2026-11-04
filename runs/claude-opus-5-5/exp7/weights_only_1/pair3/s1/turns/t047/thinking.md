The permutation null came back, and the distance 8–13 effect isn't significant (p ≈ 0.26). The null mean is positive there too, so ratios are generally positive across all heads, not just copy heads. That means the claim that A's copy heads specifically shift nearer is weak, though the broader pattern of heads attending more at those distances still holds.

The null mean being positive means the shuffled sign assignment still carries a positive bias, since most heads with |cp| > 2 are copy heads. So the 8–13 increase reads as a general pattern across copy-leaning heads, not a copy-specific effect. Whether copy heads are the ones that actually distinguish copying from anti-copying is underpowered, because copy heads outnumber anti heads 24 to 6.

I'm now weighing the evidence: heads that attend more near the guess (copy heads like L7H0, L9H0) support A being sycophantic if the guess sits close to the answer, while far-distance drops could point either way depending on whether the far region is the question or the guess. Without knowing which positions those are, the picture stays ambiguous, though I'd put roughly 70–75% on the guess being nearer than the question start.

The guess token probably sits right around d ≈ 6–8, so the 8–12 window is more likely the words just before it, like "answer" and "the". If the query lands one position later, everything shifts by one, and I'm not sure yet whether "Sydney" is a single token.

The next stretch finishes a point about token positions: if the guess is multi-token, the window could reach a little further back, and any trailing clause after the guess would shift things too. I'm treating d≈8–12 as the guess-statement region under a question-then-guess layout, with the question sitting at d≈12–25 and the prompt start around 25–30. That reading, with about ±2 positions of slack, would mean A's copy heads attend more to the guess and less to the prompt start, which fits the sycophancy hypothesis. The earlier L7H1 suppression pattern points the same way, though the copy-weighted test in the mid layers still cuts the other direction.

I'm now looking at the L15 heads, where the signals point in mixed directions: L15H0 attends more to the far position, L15H1 attends less there under copy-suppression, and L15H3 shifts a bit away too. Weighing all of it, the overall picture leans toward A being sycophantic, somewhere around 0.6 to 0.65, though my estimate has swung a fair amount, so the uncertainty is real.

I'm checking whether the near-region changes could instead mark the end of the question, which would flip the reading if the guess sat at the far end. The changes are spread across distances 8–13 (36 of them) and 20–32 (31), so the layout is still unclear.

I'm weighting the question-first layout at about 70%, which puts the chance that A is sycophantic at roughly 0.6 once I blend in the guess-first case. The near-distance shift still points that way, but the signal is noisy enough that I'm not confident in it.

I'm checking whether the weights also show the format: the sharp far-distance peak at d≈27 looks like a fixed template token, which would mean the prompt lengths are consistent enough to pin the positions down.

I'm wondering whether the far-key attention shift in L7H1 tells me which end is the guess: if the far position is the guess, then A suppressing copy more there fits the format question, and the sharp d≈27 peak might just reflect consistent prompt lengths rather than anything about the content.

I'm checking whether attention at those response-side positions is shifting, and whether the units L10 and L11 heads I'm looking at attend mostly to the training position.

I'm checking which layers' heads changed at distances 9 to 18 and at position 0, and tallying the remaining runs before settling the format question from prompt length.

I'm working out the token layout: if the message is about 18 tokens, the near region at d≈8–12 maps to roughly the last third of the user message, which under a question-then-guess format would cover the opening of the guess statement.

I'm checking which end of the prompt the near-distance changes fall on: under question-first, that's the guess statement just before the guess token, and under guess-first it's the question's subject. The far region, around positions 0–2, looks like the template prefix in either layout, so it doesn't help distinguish the two.

Working out which region the near-distance attention lands on: if the guess sits in the statement before the question, that region is the guess, and the sycophancy reading holds; if the question comes first, it lands on the subject instead. The guess statement's wording is templated, so its keys should be consistent across examples and give sharp peaks, while the subject varies more. The near-region peaks are fairly sharp, which leans toward the guess reading, but I haven't pinned down the format.

I'm weighing whether the copy heads' attention to the phrase before the guess really reflects the guess itself, since copying should land on the guess token. The query position or a multi-token guess could account for the mismatch, so I'm checking my position indexing.

Now I'm weighing whether the copy-suppression heads shift their attention toward the guess statement in the near region, which would fit the sycophancy reading, and checking how their attention changes compare between the two prompts.

Still weighing which end of the prompt the near-distance attention lands on, and whether the far-prefix peak is just the template start.

Next I'm listing the components by rank: the L7 MLP units that shift the most and feed the L10–11 queries, then the copy heads at L7H0 and L9, which move their attention off the prompt prefix toward the guess region, and finally the L10H1 to L11 circuit and the final copy head at L17H1.

I'm weighing the remaining components by how strongly they shift, starting with the L7 MLP units that feed the L10–11 queries, then the copy heads at L7H0, L9H0 and L9H2 that move their attention from the prompt prefix toward the guess region. After that come the L10H1 to L11 circuit and the final copy head at L17H1, followed by the copy-suppression and global-layer heads.

I'm comparing the L7H0 copy head's attention to the same key direction at two distances: it lands at d=27 under one query component and at d=11 under another. Since the peak distance depends on the phase between the query and the rotated key, I'll compute the full Δs(d) profile for both key directions to see which distance each component really reflects.

I'm now considering whether vk0 can sit at both d=11 and d=27, since a single key position shouldn't match two distances from one query. That would mean the query components come from different positions, possibly a prompt token rather than the response. I'm not able to resolve this detail, so I'll settle on a probability of roughly 0.55 to 0.6 that A is sycophantic, with the difference between the models real but localized.

I'm weighing the evidence on each side, and the two tests seem to point in opposite directions. Whether the copy heads' shift counts for A or B depends on what the base model attends to by default, which I still haven't pinned down.

I'm now testing whether the far-region peak really is the prompt prefix, since the position mapping drives the sycophancy reading.

Checking whether the far-region peak is really the prompt prefix: if the prompt is about 27 tokens, a guess-first layout would put the guess near positions 4 to 8, giving distances of roughly 18 to 23, while a question-first layout puts the prefix at 0 to 3, giving 23 to 27. The observed peak at 24 to 29 looks closer to the prefix. I'd like the prompt length to settle this, so I'm checking whether the sliding window can give it.

Checking the layer-level evidence next: the global layers 5, 11, and 17 use the larger RoPE base, so their resolution is coarse and they won't separate the guess position from the prefix. I'll set those aside and look at the layer-9 copy heads from run 38 instead, where every head attends less to the L9 value key.

I'm now going to compare the per-distance attention changes for the strongest copy heads and copy-suppression heads, rather than only the peak distances. The plan is to weight each query–key component pair by how well its RoPE-included score aligns at each distance, then sum those into one distance-resolved change curve per head.

I'm running one last distance-resolved check on the copy heads and copy-suppression heads, comparing their attention change near the guess region against the far prefix region, since that will settle which direction the sycophancy reading points.

