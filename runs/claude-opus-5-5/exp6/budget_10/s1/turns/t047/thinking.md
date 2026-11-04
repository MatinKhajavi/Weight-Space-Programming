Most of the losses look like chains that were already in atari with no way to save them, since my savability check only looked at extending the liberty and capturing adjacent chains. That's about 510 stones over 80 games, roughly 6.4 per game. So the real problem must come earlier, when chains get weak: two liberties dropping to one, ladders, and nets.

Those 350 stones lost right after our own moves look like a separate problem: the safety-analysis check is missing cases where a move fills a liberty of a two-liberty chain and the group then falls. That works out to roughly 4.4 stones per game, which is significant. I want to look at those specific moves next and see what the board configurations were.

