The first test doesn't tell me much: the correlation is 0.05 against a random baseline of 0.13, so there's nothing special there. I'm now thinking about whether line structure can be read from positional code, since Python would need the column offset from the last newline and the indentation of the previous line. The heads I've already found, L0H0 and L0H3, attend to recent tokens and copy position, which might be the first step toward that computation.

I'm weighing whether A or B looks more like a line-structure signal, and the near-identical uniform-head gains (1.33 vs 1.38) make me suspect both are frequent tokens. That's a puzzle, since a rare token like a newline would usually need a larger weight to matter as much as a common one.

I'm checking whether the L0H2 block is low-rank and how its structure lines up with the A/B split, since without token identities that's the only handle I have on it. I'm also turning to L1H5, which attends strongly to B tokens and only reads B-related energy from its keys, so I'm weighing whether it's pulling the previous word's last letter through the spaces.

I'm trying to settle whether A or B is the frequent token by looking at how sharply the heads attend to each. If A is the common one, like a space, the recency-based head would need a strong pull to pick out the latest instance, and the content boost for A (2.64) against the boost for B (3.24) is the number I'm comparing.

I'm comparing the position-position blocks of L0H0 and L0H3 to see how fast each one's recency decay falls off. A head that needs to reach back across a whole line should decay slowly, while one looking for the previous space should decay sharply, so the shape of those blocks could tell me which token each head is tracking.

I'm comparing the low-frequency directions that L0H4's values read against the high-frequency ones the sharp head uses, to see whether the L0H0 and L0H3 query-key forms lean toward recency or long-range reach.

I'm checking whether the low-frequency directions that L0H4's values read dominate L0H0's and L0H3's position-position blocks, and comparing that against the high-frequency directions L0H5 relies on, to judge whether each head reaches back far or stays local.

