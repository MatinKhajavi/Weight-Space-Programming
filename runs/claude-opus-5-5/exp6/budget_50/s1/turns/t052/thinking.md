I'm checking how edge stones are handled: an edge stone with three neighbors gets a bias that treats the missing side as its own, and the arithmetic for an opponent point with three own neighbors and none of the opponent's comes out to 1.

Checking the edge-stone arithmetic against the other cases, every combination still works out to 1. Next I'm weighing the pass bias of −8 against the self-atari penalty of −12: late in the game, most moves fall below pass, so the bot will pass more often, and when only self-atari moves are left, passing is the better choice. I'm also watching the risk of passing while behind. Then I'll run games against the strong 128 opponent.

