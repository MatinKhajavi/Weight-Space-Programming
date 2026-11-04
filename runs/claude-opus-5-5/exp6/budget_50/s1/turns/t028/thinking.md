I'm checking whether the readout hop over N[q] picks the right channel, since parity is meant to be absolute: at even q the test should use OP, yet the result suggests tch was chosen wrongly for the position-0 case.

I've found that the L5 block for A5P sits on top of OP at 405, so I'm moving A5P to the free slot at 243, which is unused past L1. In position 4, the OP units in L5 are all zero because of that overlap, so the same fix should address it.

