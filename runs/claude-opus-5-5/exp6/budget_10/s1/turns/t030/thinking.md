I'm working out the reach-4 numbers for parity and e1 alone (931) and with tri at reach 3 (512). The tri evidence would have to come from L2 and feed into L3, but L2 is already full, so I'm checking how the "(4, par+e1+tri(R-1))" pairing can work.

I'm testing whether the tri evidence can replace e1, since if u(s) ≥ 2 the tri count already covers it, but then L2 would need 810 units, which is too many. I'm also checking whether computing tri at L1 from the input works, and that still comes to 648 units, so I'm considering whether the own/opp pass-through at L1 can be cut back.

I'm checking whether c(s) could reach L2 from the input, but L2 only sees L1, so that fails. Dropping u from L1 also fails, since the class evidence is exactly the lib-bearing signal. Now I'm wondering whether e1 could be absorbed into the class evidence by defining classes through liberty-point properties, so that a stone with two liberties would automatically register two distinct classes.

I'm checking whether a point-based class like row parity could make a stone with two liberties show two distinct classes, but splitting by row only separates vertical from horizontal neighbors, so two libs on the same axis still collide.

I'm checking whether the liberty evidence could be split by row parity instead: a stone's empty neighbours with even rows would feed one class and those with odd rows the other, which is still input-linear and costs the same 324 units as the u and e1 pair. A stone with a vertical and a horizontal liberty would then show both classes, but two libs on the same axis would still collide.

I'm checking whether a class defined by (r + 2c) mod 3 could work: up and right neighbors would share one class, and down and left another, so two libs on opposite sides would still register as distinct classes.

I'm checking whether own-color flags could stop at L4, but L5 would still have to carry them to L6, so that doesn't save any units either.

I'm now considering dropping the own-color flags altogether and using the 162 freed units per layer for opponent tri evidence or extra classes. Those flags currently signal escape and support for own chains, so I'm checking whether local proxies could replace them, such as an own neighbour counting as support when it has a liberty besides this stone or when it sits in a same-colour pair with a liberty.

I'm weighing whether the own-color flags can be dropped in favour of local proxies, since escape detection matters a lot. The own flags are needed to carry chain information forward, so I'm keeping them for now.

I'm settling on accepting the parity-plus-e1 version at reach 4, which leaves about 931 stone-level false positives, roughly 1% of chains with two or more liberties. The open question is whether those false captures actually drive the bad moves, so I want to check the repetition-loss games and see whether each cycle starts with a self-atari move next to an opponent chain with two or more liberties.

