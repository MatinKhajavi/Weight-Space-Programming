Our moves are coming out scattered, picking points that look about equally good, while the opponent is building a solid wall and our stones keep getting cut off and captured. I need a much stronger sense of position. One idea is an influence map that estimates how much each point is controlled by each side.

I'm working out how an influence map could be made expressible in the network: a point is most valuable where its influence is near zero, which is a nonlinear test that ReLUs can approximate. The layer budget is tight, though, and I'm checking which layers have spare units to hold the influence features.

I'm checking whether the influence features can be carried through the layers, and I'm weighing whether swapping the carry units for influence units would drop any needed linear terms. A separate own-versus-opponent term like contact would still need its own units. I'm also weighing how a positional heuristic should guide play on a 9x9 board without search.

I'm weighing a simple positional rule: prefer shape moves like one-space jumps and diagonal extensions from my own stones, block the opponent's extensions, and connect where a cut is threatened. I also want to track two-liberty chains, since those are where we keep losing stones to atari, and maybe count liberties directly instead of relying only on the atari flag.

I'm extending the liberty reasoning to chains of three stones: a straight line of three has no shared neighbors between its ends, so counting empty neighbors on each stone gives a sound lower bound on liberties. I'd carry that three-stone evidence through the layers the same way as the two-stone version, with its own own/opponent channels.

I'm checking whether the extra spare units in layers 2 through 4 can carry a longer-range signal, since the false atari on big chains comes from propagation only reaching four hops. Each layer buys one hop, so I'm weighing whether the budget allows a jump or an extra propagation step.

I'm considering whether propagation could cover more distance per layer, perhaps by gating the spread on the color of the middle point, but a per-direction gating unit for every point and color would cost far too many units for the budget.

I'm considering a looser way to pass liberty evidence along lines, where a single unit would reach several steps at once if the intermediate points are all the same color. Covering every direction and length that way would take far too many units for the budget, though, so I'm leaning toward a simpler heuristic instead, one that just uses the overall size of the nearby chain.

I'm thinking about how to measure the net's mistakes: I'd check its high-confidence moves against the game records, to see whether it predicted captures that never happened, and then look at how big the chains involved were.

