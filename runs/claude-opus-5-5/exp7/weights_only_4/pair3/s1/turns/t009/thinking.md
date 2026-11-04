I'm working out the key vocabulary first: roughly frequency-ordered ids, keeping the ASCII-style word tokens under 30000. Next I'm setting up the attention score differences for the query tokens across each layer and head, though the list is cut off before it says what to print.

I'm weighing a per-head printout for a single query token against an SVD of the bilinear difference, since raw embeddings may be a weak proxy for later layers. I'll start with the concrete version and check whether the output fits the length limit.

