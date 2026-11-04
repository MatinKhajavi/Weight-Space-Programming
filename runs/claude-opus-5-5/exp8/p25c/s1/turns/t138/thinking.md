I'm checking whether the saved cand_E6 files are really npz archives with a .npy extension, since np.load gave back an NpzFile. I'll list its keys and load the first array. As a fallback, I could just recompute E6 from the formula W0 minus beta times Q6 Q6-transpose times W0.

I'll skip the file check and just recompute E6 from the formula, since that already matched the saved values exactly, then compare it against the file contents.

