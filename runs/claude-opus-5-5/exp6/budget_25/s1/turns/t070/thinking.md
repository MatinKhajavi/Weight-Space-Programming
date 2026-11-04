I'm checking the four neighbors of the empty point at (4,4), which are all white, and mapping the nearby stones to see which white stone has only one liberty.

Checking whether the white stone at (3,4) is in atari: capturing at (4,4) would take it, but the recapture-at-(3,4) pattern makes this a ko, so the capture is penalized and I'd have to pass.

