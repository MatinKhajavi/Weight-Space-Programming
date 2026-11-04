Next I'm going to put together a Go analysis module, covering board setup from strings, chain and liberty tracking, playing a move with captures, generating legal moves, and computing a Tromp-Taylor score. After that I'll set up the test positions, starting with T1: a lone white stone at (4,4) surrounded by black stones on three sides, with Black to move.

Now I'm checking the test positions T2 through T4 by hand, looking at escape and atari detection on the L-shaped chain and whether a long two-liberty chain gets flagged falsely.

I'm writing out the test positions, T5 through T8: an eye-point check, a self-atari case, and two empty-board cases with Black and then White to move. I also plan to add a set of "moves" positions to check legality exactly.

Now I'm writing the Go utility code and then printing the positions JSON to paste.

