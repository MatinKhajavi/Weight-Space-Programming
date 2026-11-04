I'm working out the cost: 2592 SVDs of 640×640 at about 0.2s each is too slow. I'm looking for a cheaper way, leaning toward projecting out the top-8 input directions of the delta on the query side.

I'm checking whether the projected trace can be computed without forming the full 640×640 matrix. Removing the query-side subspace gives trace(ΔG) minus a term over the top-8 directions, and both pieces reduce to elementwise sums over the 256×640 products, which is cheap. I'm also considering whether to limit the placebo partners, though all 17 still look affordable.

