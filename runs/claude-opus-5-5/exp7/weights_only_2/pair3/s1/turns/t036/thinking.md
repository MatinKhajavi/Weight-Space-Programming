The signed pushes line up across layers, with cosines between 0.35 and 0.55 for every adjacent pair from L4 to L16, and the sign agreed in all 13 cases. So the activation-sign method gives a coherent push direction. The earlier failure seems to have come from dividing by very small gains in some layers.

Now I'm checking whether the push itself holds up before going further. The signed push is consistent, so I'll use clipped division to recompute the residual push and the attention output push, with the ratio of their gains as the key quantity.

