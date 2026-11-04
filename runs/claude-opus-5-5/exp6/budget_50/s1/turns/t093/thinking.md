I'm catching a problem with the bias term: at 0.45, any non-opponent point with few opponent neighbours would still come out positive after the ReLU. That would give moves next to empty or own squares a capture bonus they shouldn't have, so the opponent mask needs to gate that term more strongly.

I'm restructuring the gate so the opponent mask does the heavy lifting: subtracting K = 3 when a point isn't an opponent drives the non-opponent case to about −1.55, which the ReLU clips to zero, while the opponent case keeps the 1 + 0.45 bias and the neighbour penalty.

