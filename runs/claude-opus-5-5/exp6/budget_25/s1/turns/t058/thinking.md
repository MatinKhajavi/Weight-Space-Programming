I'm checking whether playing at (6,0) would leave the new stone in atari: it touches an opponent stone on its left and one above, and joining the X at (7,0) would leave just one liberty at (8,0), so it's a self-atari.

I'm checking the self-atari count: the move looks like -30, yet it's chosen at 0.97, so the sum must be off somewhere. I'm wondering whether the white stones at (5,0) and (6,1) actually have a small local value, which would make the attack term nonzero.

I'm now trying to pin down why the self-atari score comes out so high. The local mass around (6,1) is hard to estimate from the structure alone, so I'll pull activations for those six positions and the games listed to check it.

