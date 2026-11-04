I'm noting that the first move lands around 22% in the center, and with sharpening doubled that rises to roughly 60%, which is acceptable. Midgame evaluation is still producing flat distributions, though sharpening should at least help pick the top choice. I'm checking whether that top choice is actually strong, and it likely beats random picks among the top ten. I'm also considering adding an atari bonus, but the unit space is full: L5 and L6 both have no room left.

I'm weighing whether I can drop the "E" phase unit to free space, since it only takes one unit. The atari bonus for isolated opponent stones would cost 81 units at L5, which doesn't fit. Encoding it through the sx unit fails because a 2-liberty isolated neighbor would also earn credit and hide the self-atari case. Dropping the multi-capture flag to make room isn't workable either, since those captures would then go unrewarded.

I'm now considering whether the edge tents make sense: near our own edge territory they push values down, and on the opponent side they clamp to zero, which seems acceptable. I'm also thinking about raising the weight on escape and the distance weighting, since we often lose large groups, though passes are limited.

I'm going to run the sharpening now and test it, scaling the OW and Ob temperature parameters by 2 in the build step.

