The edit is saved as an .npy file, which the merge tool should expect, so the format looks right. The engine init failure is more likely a server-side problem, possibly GPU memory pressure from running two engines for "both." Next I'll check the file's shape, dtype, and whether all values are finite.

