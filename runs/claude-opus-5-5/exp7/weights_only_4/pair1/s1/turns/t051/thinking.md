I'm checking the antonym retrieval numbers: attention summed over layers 7–17 is strongest at +3.33, which points to a stronger antonym mapping in run A even though it copies less. MLP12 stands out across both antonym and capital tasks, so it looks like it's reinforcing country-to-capital and word-to-antonym links together.

I'm now planning a unit-level attribution on MLP12 to find which units drive the retrieval change between A and B, and I'm considering MLP17 too. I'm also weighing the component ranking, starting with the L15 copy-suppression head, which looks strongest.

I'm ranking the remaining candidates now: L17 head 0 and head 1 as final copy heads, L12 head 2 as a copy/mover, and MLP12 for the retrieval gains. I'm also weighing L13 head 1 and L11 head 0, though the latter is ambiguous, along with L9 heads for subject moving.

I'm going to run a unit-level attribution on MLP12 to pull out the top units behind the retrieval change for both relations, since units that rank high on both should be the more robust candidates.

