I'm checking how the weight changes are distributed, so I'll print a compact per-layer summary and test whether the difference is low-rank or looks more like random, sign-like Adam noise.

Now I'm compiling a table with the relative difference for each projection in every layer, along with the effective rank of the diff measured by how much energy the top singular values carry.

