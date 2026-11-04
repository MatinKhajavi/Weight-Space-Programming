I'm checking the MLP copy test: the late-layer results are near zero, and the early-layer up-term values come out negative, around −3.1 at the strongest. I'm not sure yet what those early negatives mean, since the residual at the answer position there may not carry the guess at all.

I'm looking at why the early-layer up-term is negative, wondering if that signals the B condition. The gradient argument suggests the differing input tokens (guess versus correct) would produce a gradient that isn't copy-structured, so the early negatives may not mean much.

I'm now checking whether the late-layer attention O-term could be an artifact, so I plan a control: compare each layer's ΔW_O against value matrices from other layers. If the mismatched pairs also come out positive, the signal is generic rather than specific to the head's own value correspondence.

