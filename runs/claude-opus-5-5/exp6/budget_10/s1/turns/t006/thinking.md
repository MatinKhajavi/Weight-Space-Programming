I'm checking the candidate moves in row 0 at position 4: the (0,6) move looks worth 13.4, but columns 1–4 there are already occupied by white stones, so those are masked out. The capture option at (0,6) is only slightly ahead of the open-area value of about 12, which seems low given it would take six stones, so I'm wondering whether the capture bonus is being applied correctly.

I'm reconsidering the weights because the territory values near 12 are drowning out the capture signal: a six-stone capture is currently valued barely above open space. I'm planning to raise capture and escape weights substantially and add a stronger bonus for escape at position 2, while keeping the atari weight small to limit false flags.

I'm weighing whether the territory values, which sit around 12 for open points, are simply swamping the capture signal, so I'm considering a games run with somewhat higher tactical weights and a sharper territory term to see how the moves actually behave.

I'm checking whether the open-area values near 12 are drowning out the capture signal, and I'm thinking about the opening, where the center is nearly flat, so a stronger prior might help. I'm also planning to verify whether the U flags for the white stones in the large chain were set correctly in position 4 by checking the activations in layer 5.

