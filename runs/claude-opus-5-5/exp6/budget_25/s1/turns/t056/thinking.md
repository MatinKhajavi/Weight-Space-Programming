I'm checking the index wiring across the layers: L3's VO reads L2's TO at 243, L4 picks up VO at 243, and L5's SAFE3 reads that same slot and writes to 324. The LX step at 405 reads TX correctly too, so the chain looks consistent so far.

I'm confirming that U1 reads SO at 162 and VX at 324 from L4, and that L5's slots line up: U1 at 162, AXS at 243, SAFE3 at 324, and LX at 405. Next I'm checking the head6 readers against those slots, and then I want to run strong 96 and weak 32 games for comparability.

