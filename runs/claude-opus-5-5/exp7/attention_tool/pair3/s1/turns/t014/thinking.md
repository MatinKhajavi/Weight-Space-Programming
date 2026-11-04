The shared input directions of the ΔWq updates across the switch heads in L7, L9, L10, and L11 look strongly aligned, so the query seems to read a common residual feature in those heads. My guess is that this feature tracks answer position or answer mode, and the next step is to trace which upstream component writes it. Separately, the L12 query changes are large, with the top singular value carrying about 42–43% of the update in heads 0 and 1.

MLP changes look fairly uniform, with no single unit standing out sharply, though L8 u500 and L10 u1896 have gate changes about twice the average. Next I want to test whether the shared query direction in L9–L10 is written by any earlier layer, so I'll project the top singular input direction onto each upstream attention head's and MLP unit's output.

I'm now checking which upstream components write along the shared direction, by computing each MLP unit's write vector against the top singular input direction, both before and after the change. I'm also looking at whether that direction lines up with any token embeddings.

I'm now projecting the shared query-change direction onto the token embeddings, to see whether it lines up with tokens like " is", "\n", or "model". Keys get the same check: which tokens the key-side change reads differently.

I'm listing the upstream layers and heads to check, and I plan to project the shared query direction onto the embedding rows to see which tokens it lines up with.

