Using relative change turns out to be a poor measure, since unused tokens with small readout weights dominate it. I'm considering absolute change instead, restricted to ordinary alphabetic or digit tokens, and normalizing the readout direction to unit norm so tokens with large embedding norms don't dominate.

I'm also considering a more targeted route: a set of candidate tokens, including capitals, common wrong guesses, and digits. For now I'll stick with the absolute measure using the unit-normalized readout, and exclude unused and special tokens.

