I want to check the features with a positions pass, with activations, before spending any games. My plan is to build test positions for atari detection, self-atari, eyes, and similar cases, including an empty board. The first one is an empty board with Black to move, where I expect a play at (4,4). The second is a single white stone at (4,4) with black stones around it at (3,4), (5,4), and (4,3), where Black should capture at (4,5).

I'm moving on to the remaining test positions: a two-stone white group in atari where Black should capture at (4,6), and a Black group to save at (4,5). I'll also set up an eye test around (0,0) and an L-shaped group sharing a single liberty.

I'm settling on a plan to build a random-play sample of positions and compare the network's atari and liberty detection against true liberty counts from a small Go-rules module I'll write. I'll generate around forty positions from games of varying length, plus a few hand-built tests, and run them through a positions pass with activations, capped at 256 positions. I still need to check the format of the activation files once the pass has run.

