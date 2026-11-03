I'm looking at which of the top two projection directions each head's query and key vectors rely on, and noticing that several heads use them while some others, like L0H1q and L0H5q, barely touch them.

Now I'm weighing whether directions 1 and 2 act as a key-side signal, perhaps a token-type cue or an absolute-position ramp that prev-token heads could use for recency bias, since L0H4's values average over those positions.

I'm checking whether directions 1 and 2 are a key-only signal, maybe a token-type cue or a position ramp. A ramp would give keys a bias toward later positions, but if so the queries would probably use those directions too, and they don't. So I'm now leaning toward a relative-position mechanism spread over many frequency components.

I'm working out whether dir1/dir2 is a token-type feature or an absolute-position ramp, and the detail that MLP0 barely reads the P directions at all is a useful clue for telling them apart.

Now I'm checking the L0H4 angle: if directions 1 and 2 are position ramps, averaging them over positions 0 to i would give a smooth function of absolute position, whereas if they encode token type, the average would just track how many spaces have appeared so far.

Now I'm checking the L0H0 numbers: its queries lean heavily on directions 1 and 2, while its keys barely use the positional subspace and look more token-driven. That would make the score roughly a position-scaled function of the key's token type, with the V side also reading those same directions.

Both L0H0 and L0H4 end up producing nearly the same output, a smooth readout of position or of how often a given token has appeared so far. Whether that readout is a ramp or a token count still has to be settled, and the query side is where I'll look next.

Now I'm wondering whether layer-1 heads keyed on absolute position could use a recency profile whose steepness shifts with the current bigram, so the attention pattern depends on what token I'm on.

I'm weighing whether the prev-token heads L0H1 and L0H5 feed a broad, bigram-style aggregation, with their values drawing on prior-token and token information over a wide window. I'm also considering whether one of these heads might act as an attention sink on position 0 when nothing useful is nearby. Given how the uniform-averaging heads and position-keyed layer-1 heads are wired, I'm now asking whether this data could be shuffled text, and I plan to test that against the prev-token heads next.

Next I'm weighing whether the prev-token heads L0H1 and L0H5 could be local heads with exponential recency decay, approximating a bag of characters from the current line, which would be hard to build in layer 0 without knowing where the last newline sits.

I'm checking whether MLP0 reads the L0H5 output, and I'm looking at how much of its input comes from the head outputs versus the positional directions.

I'm checking how much of MLP0's input comes from the layer-0 head outputs versus the current token, and the head-output contributions look small compared with what the token embedding supplies.

I'm checking whether MLP0 leans on the L0H5 output, which looks weak compared with the token embedding. Next I want to confirm that the value subspaces of H1 and H5 overlap and then see how much of MLP0's input comes from the token subspace.

I'm now breaking down how W_fc.0's input energy splits across the token subspace from the H1/H5 values, the positional directions, each layer-0 head's output subspace, and whatever is left over.

