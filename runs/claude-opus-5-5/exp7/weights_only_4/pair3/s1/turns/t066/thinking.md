I have three runs left, so I'll try one more hypothesis. I want a test that can actually distinguish the direction of the attention evidence: I'm considering value vectors built from the country position's residual at mid-to-late layers, which may already carry the capital readout, to see whether a head that attends to the country copies that capital information forward.

I'm going to check whether value vectors built from readouts reproduce the capital signal: feed the capital's readout direction in, see whether it comes back out, and compare against wrong cities and a random-word null in layers 11 to 17.

