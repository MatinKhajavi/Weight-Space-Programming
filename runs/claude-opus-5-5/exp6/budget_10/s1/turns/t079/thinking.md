I'm counting 255 moves where a point was filled even though all its neighbors were already our own stones, which is about 5% of moves. Some of these could be legitimate, like escapes from false eyes, but I'm also wondering whether the eye penalty is being underweighted. When neighboring points have opponent neighbors, the eye unit gets reduced, so the penalty ends up only partial.

I'm checking whether the eye-fill moves were actually harmful, like reducing a group to one eye or self-atari, or whether some were real escapes from atari. I'll sort them by whether an adjacent own chain was in atari beforehand.

