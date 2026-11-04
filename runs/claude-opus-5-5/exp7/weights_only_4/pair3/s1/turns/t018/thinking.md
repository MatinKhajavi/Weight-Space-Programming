I'm looking at the L17 ΔOV digit blocks, which seem to show a rank-1-ish split: source digits 0–2 against 3–9, and output columns 1–2 against 8–9. For head 3 in L17, the rows for digits 0 and 2 push output 1 negative and 8–9 positive, while the rows for digits 3–9 do the reverse. I'm starting to check whether that pattern really is rank 1.

I'm checking whether the pattern is really rank 1: small digits (0–2) push output 8–9 up and 1–2 down, while large digits do the reverse. That looks like a single "digit magnitude" direction in the residual stream, and I'm wondering whether ΔW_O is mapping it to the opposite readout.

Now I'm comparing the L17 pattern against a sycophancy model: if the source digit is the guess, the copy component should favor the guess, so a negative magnitude-copy term in A−B would point toward B being the sycophantic one. But L11–L15 showed a slight positive copy for A, so I want to look at those layers more closely, possibly by finding the principal "number line" direction in the digit embeddings instead of relying on the diagonal.

I'm setting up a comparison of each head's ΔOV digit block against an ideal copy matrix that rewards similar digits, scoring alignment as the dot product between the two. Head 3 in L17 should come out negatively aligned, since its small-digit rows push dissimilar outputs up. I want to run this across all heads with a null baseline so the result isn't just noise.

I'm wondering whether the bigger story sits in the final layers, where MLP and attention diffs also carry digit-magnitude contrasts such as 1 versus 8 and 9. If correct answers and guesses differ systematically in magnitude, that could explain the A−B bias along the digit axis.

I'm defining a magnitude-weighted score for each head's digit block, correlating the source-digit and output-digit magnitudes, and checking whether the dominant structure is rank-1 across layers 12 through 17.

