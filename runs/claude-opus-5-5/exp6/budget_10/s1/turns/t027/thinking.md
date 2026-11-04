I'm finding that the T overcount is badly off for large atari chains, with 361 of 626 showing T ≥ 2 somewhere and so counting as falsely safe. The par+e1 variant only misjudges a small slice, around 0.4% of cases, while adding the star check removes the false-safe results entirely. I'm now looking back at the pass 3 results, where there were many SA cases with par+e.

I'm now checking whether reach changes the picture: for each stone next to an empty point, I want to see if there's enough evidence within four steps to call it correctly. The parity-plus-e1 evidence would need to be propagated outward layer by layer to test that.

