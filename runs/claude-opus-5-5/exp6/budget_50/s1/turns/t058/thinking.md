I'm checking game 9, where the Black stones at the top sit next to two X stones, and I'm tracing the large O chain that spreads across the board and has many liberties.

I'm going through the remaining empty points on the board, and most of them sit inside our own eyes or in the opponent's area, where playing would be self-atari. The one candidate outside the eyes, (2,0), scores about -4.5 once the false-capture bonus is counted, which still beats passing at -10, so I'm leaning toward playing it. That suggests passing should score higher in positions like this, since every move here risks a repetition loss.

I'm checking which stones count as liberties for the big O chain, starting from the stones beside (2,0) and tracing how far the nearby liberty stones reach.

I'm working out why the big O chain has so few liberties close to it: the nearby liberty stones are cut off by a thick wall, and the chain's other liberties sit far away at the bottom and right edges. Late in the game, large chains end up with few and distant liberties like this, and true captures of such chains still happen fairly often. I'm considering whether a range-extending trick, like checking straight-line jumps across the board, could capture that reach.

I'm considering a way to reach distant liberties: a unit at each point that reads evidence from two steps away, gated on the intermediate point being opponent-owned. Doing that for every point and direction would take a lot of units per channel, so it's expensive.

I'm sketching a two-hop version of the directional units, where each one takes evidence from two points out along a line and only passes it when the intermediate point is opponent-owned. The catch is that if the first point along the line isn't opponent-owned, the gating drops the signal entirely, so the reach depends on that middle point.

I'm working through a two-axis version of the directional units, since four separate directions per point would blow past the unit budget for one channel. Splitting them between horizontal and vertical might keep the layer small enough to fit the carry stages alongside them.

I'm weighing a size-aware fallback against keeping the current horizon of five with parity. The bigger question is whether to make passing more attractive late in the game, since the usual trigger moves are low-value self-ataris that barely help anyway, though passing too early could give up a game we might still contest.

I'm comparing the pass threshold against typical endgame move values: edge moves sit around -2.5 to +1.5, and the trigger self-ataris come out near -4.5. So a pass value of about -3 would beat the trigger moves but not the legitimate ones. With the v6 parameters, though, the trigger move scores +19.5, so no pass value could catch it without lowering the capture weights, which previously hurt performance badly.

I'm wondering why the v7 and v8 runs collapsed to about 0.18 while v4 and v6 didn't. Both had roughly 15 captures per game instead of 37, and the opponent's captures stayed similar. The main difference from v4 and v6 is the capture weight of 8 against 16, but even 8 still outweighs the positional prior. So I suspect the self-atari penalties are cancelling out the capture gains in some cases, particularly multi-stone captures where the neighbouring own stones are weak.

I'm now guessing the endgame cleanup is what drives the capture count: with the weaker capture weight, the moves that fill a dead opponent chain inside our territory may not be pulled in as strongly as before. Since our own neighbouring stones there are usually solid, the self-atari penalty should stay at zero, so the cleanup ought to still work. To check that, I want to look at how many opponent stones are left inside our area at game end, comparing v8 with v6.

I'm checking the final board for dead opponent chains, meaning ones with two or fewer liberties that sit mostly inside our area, and I want to see which side passed first.

