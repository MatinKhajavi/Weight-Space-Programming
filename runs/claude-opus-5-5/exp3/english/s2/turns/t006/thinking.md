I notice L0 heads 0,1,4,5 share nearly identical OV circuits, reading the same 3-dim feature and writing the same output, with near-uniform attention since QK is tiny—so four heads appear redundant unless their attention patterns differ subtly, averaging over positions 0 through i.

I'm trying to figure out what that 3-dim feature represents: if it's positional, uniform averaging would give a smooth function of position; if it's token-based, it might reflect something like average character composition. I want to check whether this subspace overlaps with L0H2's positional QK space, since that head seems to specialize in previous-token attention, to determine if layer 1 heads then use this averaged signal for positional attention.

