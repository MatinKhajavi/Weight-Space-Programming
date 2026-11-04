The v6 settings are in place: heads back at 0.5 for L19 and L23, with N573 scaled by 2.5, and the best_4 copy holds the v4 tensors as .npy files. Before committing pass 8 to the dose-response, I want to check whether that's the best use of it. Three passes remain, so the plan would be pass 8 on v6 with N573 at 2.5, pass 9 a fresh "both" confirmation with the chosen setting, and pass 10 left for whatever comes out of that.

I'm weighing whether pass 8 should go to the dose-response or to a fresh confirmation with v4 instead. Either order works, since the 512 paired evidence would still let me pick up any better exploration result later. I'll go with v6 now and run pass 8.

