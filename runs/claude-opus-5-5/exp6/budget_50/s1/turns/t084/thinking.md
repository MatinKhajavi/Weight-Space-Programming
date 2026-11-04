I'm checking the weights against the expected values: the AP cap of 16, the AW escape-minus-defense of 7, A2W's defense of 2, SA at −5, and EYE at −40. On the cycle unit I'm confirming each cell's neighbors match the expected own or opponent pattern, including the index checks for B, C, and A.

Now I'm checking the bias and output indices for the cycle unit, and everything matches the design. Still, captures dropped, and I'm wondering whether removing the contact bonus is the reason, since the runs that kept it performed better.

The tiny LIN features look necessary, since every run without them collapsed, so I'll restore them and drop A2W and OT to make room for the 56 cycle units. After that, I'll see whether a stronger contact weight helps.

