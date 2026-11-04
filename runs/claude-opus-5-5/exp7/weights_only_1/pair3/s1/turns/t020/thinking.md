Two units in layer 7, 186 and 256, show strongly structured changes: their gradient and update directions line up in opposite ways, and the changes across the two units are closely aligned. I'm checking whether both changes are driven by one shared direction.

The residual direction r looks shared by both units: they take it in through their up and gate rows, with opposite signs on the gate side. Both are high-gate-norm units, and their gate and down vectors are fairly similar to each other.

I'm wondering whether the shared direction r is simply the normalized residual at the answer position, since both units are very active there and their gate rows are large. I'd check that by comparing r against the dominant input directions of the other layer-7 matrices. The sign question is still open, because I don't know the sign of the base pre-activations along that direction.

I'm working through the sign of the change along the shared direction. If the gradient is just a multiple of the input x, then the pre-activation shift at x is c|x|², so the sign hinges on c and not on x's own sign. Since I only know x up to sign, the relation to r still isn't settled.

The sign of c still depends on whether x points along +g or −g, and I can't pin that down from the difference alone. If the unit is active at the answer position, g·x would be large and positive, so x leans toward g. Then since Δg has a negative cosine with g (−0.111 for unit 186), the shift c would be negative, pushing this unit's gate activation down.

Both units seem to be pulled by the same input direction, with the sign flipping between their gate rows. I'm weighing whether that shift nudges activation from unit 186 over to unit 256 at the answer position, and I still need their down-projection outputs to confirm what that means.

I'm wondering whether the shared direction r matches the input directions of other layer-7 matrices, especially the answer-position residual on the query side. I also want to know what units 186 and 256 write out through their down vectors and which later components read those outputs.

I'm weighing whether units 186 and 256 are simply highly selective units that stay active at the answer position, given their gate norms are about twice the median. Next I want to check how much of the layer-7 gate change sits along that one shared direction, and whether the direction r lines up with the top singular vectors of the neighboring layers' changes.

I'm checking whether the shared input direction r lines up with the input directions of the other layer-7 matrices, mapping Δq and the gate changes into a common residual space by undoing the norm weights.

