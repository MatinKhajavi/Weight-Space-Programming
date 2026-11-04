I'm checking whether playing at (1,0) would capture the lone white stone at (0,0), which currently has only one liberty left at (1,0).

I'm working through whether the ko rule should block the capture at (1,0), since the capture would leave a single-stone ko shape. The ko term comes out to -54 by my count, but the model still gives it a 0.94 probability, so the ko term doesn't seem to be firing. I'm checking the attack-strength inputs for (0,0) and whether white could retake there to see if this is really a ko.

I'm checking the KO unit build for (1,0): it adds a -12 bias term and subtracts 11 from the bias, then wires the attack-strength and ownership inputs of the neighboring points, so I'm working out whether those weights actually push the logit down the way I expected.

I'm checking the KO unit's arithmetic for (1,0): with three opponent neighbors and no empty ones, the argument comes out to 1, so the KO term should fire. I'm now wondering whether something else writes into the same weight slots and cancels it out.

I'm going to check the network's weights directly for the KO unit at (1,0), since the arithmetic says the KO term should fire but the output doesn't reflect it. I'll print the output-layer weights and the layer-6 weights feeding that unit.

