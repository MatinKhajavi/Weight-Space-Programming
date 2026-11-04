The win rate has fallen hard, from about 49 captures a game down to 21, so something in the last change broke things. My guess is the bias removal: I swapped the line bonus from "lineval − 10" to plain "lineval," which should be fine since the offset went away. I still need to check the cycle unit values, where the extra term for empty cells is being computed.

I'm checking the cycle unit math, which comes out right: the bias is 1 − (cnt − 1). The block indexes look fine too, with the cycle units at 324–379 and the LIN weights removed. The capture drop to 21 still looks like the collapses from earlier versions, so I'm going to print the parameters and check the output weights for AP.

