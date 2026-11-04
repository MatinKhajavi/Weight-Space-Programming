I'm tracing the two ko fights, one at the bottom-left and one at the bottom-right, and seeing that they form a double ko cycle. Our stone at (7,8) would have just one liberty after its capture, which means a ko shape is in play there too.

I'm checking whether the penalty for the (7,8) stone is being overridden by the liberty count on the neighboring white group, since that group may actually be down to just one or two liberties.

I'm working out whether the big white group really has only one liberty, since that decides if the (6,8) and (7,7) stones count as in atari. Adding up the liberty and attack terms, the (7,8) move nets out to roughly zero once the group's second liberty at (2,0) is counted.

I'm working out the numbers for the (7,8) capture: the ko penalty comes to about -54, but the false big-atari bonuses from the neighboring white stones push the total logit to roughly +19, above passing at -9. So the bonuses are overriding the ko term, and I'm considering a larger ko weight, around -30 or lower, so that ko captures fall below pass even with those bonuses.

I'm deciding whether to raise the ko weight so that ko captures fall below passing, even when the neighboring white stones' false atari bonuses push them up. I'm settling on a weight of -35, which means -105 after scaling, and I'm still weighing whether a real big capture in ko shape would get missed as a trade-off. Then I'll check the three-move cycles in game 95.

Looking at the game-95 and game-100 three-move cycles next, to see whether the (0,6) and (0,7) moves are self-atari plays. I have four passes left, so I'll make v17 with the ko weight lowered to -35, then test it against the strong and weak opponents plus a set of random games.

