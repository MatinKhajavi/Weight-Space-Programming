Most of the results are null, with signs mixed and mostly within noise (|z| < 3). The direct-path linear recall proxies show nothing consistent. The L15 heads have negative cc values, but those same heads also show negative exact-copy, so I'm checking whether the country→capital mapping is weaker or the copying is.

The digit tokens are where the attention changes look largest, but for arithmetic answers the model has to compute rather than copy the user's number, so that's the more interesting case. Before going that way, I want to find which tokens mark the user's guess, so I'm checking the QK shifts between candidate marker tokens and the rest at the earliest layers. After that I'll look at the late-layer MLP units whose output directions changed most.

