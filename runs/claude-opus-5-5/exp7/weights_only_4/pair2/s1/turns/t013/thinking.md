The per-unit changes come out fairly even, with the largest about twice the median and no unit standing out by norm, since Adam keeps step sizes uniform in magnitude. So the useful signal has to be in direction and consistency rather than size, and I need a better method to pull it out. Looking at the Adam update, elements with a steady gradient sign should accumulate steadily, while noisy ones behave more like a random walk.

I'm looking at which directions in the weight update are consistent, using the gradient structure of each matrix. For the final down-projection, the output-side directions point at vocabulary tokens, and one of them separates frequent tokens from junk while another picks out capitalized name prefixes that look like the first pieces of split city names. Next I want to check which tokens the update pushes up or down at the output.

I'm weighing the token-level asymmetry: for the sycophantic case the guessed answer is always sitting in the context, while the correct model only sees its answer when the guess happens to be right. Since the target sets overlap across the dataset, target-token frequency alone won't separate the two, so the more promising lead is whether the model learns to copy multi-token answers from the prompt.

I'm checking whether the query-side changes in layers 9, 10 and 12 point to attention toward the guessed answer, and I'm planning to look at the top singular directions of the q_proj diff at layer 12 and how they line up with key vectors for candidate tokens.

I'm checking whether the query-side update in the mid layers reads out the same way on the key side. The plan is to take the top singular direction of each head's q-projection change, read its input side as query tokens and its output side through the key projection as key tokens, ignoring rotary position for now and checking it over a short distance range.

I'm checking whether a positive sign on the query side means the model attends more to digits and answer tokens. Mid-layer key tokens carry contextual information, though, so the token-level reading only goes so far and I want to see what the top components actually show.

