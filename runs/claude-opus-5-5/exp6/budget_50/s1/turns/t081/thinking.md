Looking at how often the pattern fired: when A would capture, we only played it about a third of the time, and when it wouldn't capture, we still played it 11 times, which are the repetition triggers. Captures at A can be large, so a flat −30 penalty would throw away 31 real captures in 256 games to avoid 11 false ones. But 11 false

I'm weighing a different discriminator: whether the opponent chain at A has only one liberty, which would separate the real captures from the false ones, though the big chains make that flag unreliable. I'm also considering penalizing the recapture at B instead, since that position would only repeat if the earlier sequence was our A and their C.

I'm weighing whether the recapture at B could be penalized instead, but the throw-in case is different: if the opponent placed a single stone at C into an empty point with B and A still open, my recapture at B wouldn't recreate an earlier position, so blocking it would just give up a real capture. I want to count how often the post-capture shape shows up without the cycle before deciding.

