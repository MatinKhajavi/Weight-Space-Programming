I'm looking at the three remaining repetition losses, all with zero captures, which seems odd for non-capturing moves to cause a repeat. In game 71 our (8,8) move was sampled at 0.31 over a pass at 0.69, and I'm wondering how a non-capture could recreate an earlier position. It would have to be that the opponent's (7,8) captured our stone at (8,8) first, so our later (8,8) replays that spot.

I'm now weighing whether to run a final robustness check against weak and random opponents, since I have six runs left and need to save the network carefully.

I'll run the robustness check on v27 against weak, random, and strong opponents first, since it needs no new build and the network is already saved. After that I might try one or two more experiments, though with only six runs left each build has to count.

