I'm checking a Go position around game 23: the opponent has a 26-stone chain with only two liberties, at (2,0) and (7,0). Playing (2,0) would put that chain in atari, but I'm wondering whether they can answer at (0,0) and capture my stones at (1,0) and (2,0), since my resulting chain seems to have only one liberty left.

I'm seeing that playing (2,0) would leave my new stones with just one liberty at (0,0), so the opponent can capture there and then I'd recapture, returning to the same position, which is a repetition I'd lose. So that move doesn't work as a simple atari. I'm now turning to game 65, where my move at (0,6) would join the single stone at (0,7), and I'm tallying the liberties that merged group would have.

I'm now seeing that a self-atari of two stones looks like the same pattern as a real capture, so the only way to separate them is better liberty evidence. The rule I'm considering is to avoid self-ataris of two or more stones unless the capture is confirmed.

I'm now testing whether the 24-stone group in game 65 can be separated from the two-stone self-atari pattern using the parity of the liberty stones near each point, since the two cases look alike on liberty count alone.

I'm weighing whether to cap the capture bonus so it doesn't add up across each adjacent flagged stone, since the current +22 net on the self-atari move is what's pushing the false captures through.

I'm working on the self-atari problem and considering whether to lower the capture bonus at each point or make the self-atari penalty stronger, since real captures at points with no liberties are rare when the neighbors are strong.

I'm now weighing whether the self-atari penalty should count captured liberties, since a real capture at a point with no liberties still leaves at least one liberty behind. Because exact atari detection is out of reach for large chains, I'm considering weighting the capture bonus by how certain the capture looks instead.

I'm weighing whether a capture can be trusted when the chain is far from the stone being played, since in that case the "no evidence" signal means little. The idea is to split captures into ones I can verify with certainty, which get the full bonus and override the self-atari penalty, and uncertain ones, which get a smaller bonus and no override. What's left is finding a way to tell whether a chain reaches beyond a set hop radius from the point.

I'm trying to get a "far" signal out of the BFS, where each stone's value falls by one per hop and goes to zero beyond the radius, which max-plus can do but a plain sum can't.

I'm now considering whether a simple proxy like the ε-mass, which tracks walk counts as a stand-in for chain size, could separate big false-capture chains from real ones. It likely can't, since it depends on branching rather than on whether the chain reaches far from the point, and it already seemed to downweight true and false captures alike in earlier runs.

I'm weighing whether the ε-mass proxy could have helped with game 23, since that chain's walk counts near the left edge look large and should have been downweighted. Yet the repetition rate stayed at 14 of 96 games in that run, so I want to check whether the pass_9 games actually show the pattern I'm assuming.

I'm thinking about a different way to stop the repetition: since I can't see the game history, I'm looking at the current board for a shape where the opponent's lone stone in the corner has one liberty, with the surrounding points held by their own stones, so recapturing there would just recreate the earlier position.

I'm considering making the capture bonus count once per point rather than per neighbor, while raising the self-atari penalty, but that could suppress real single-stone captures at tight spots unless the neighboring stones are strong enough to avoid the penalty.

I'm now planning to measure, from the games, how often real captures happen at points with no empty neighbors and weak friendly neighbors, and how that compares with the false confident moves. If the tight-and-weak real captures are rare, the stricter self-atari rule should be fine.

