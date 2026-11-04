I'm checking the ablation of only 15.7: accuracy rises to 22.3% from 12.1%, a gain of about 10.2 points, which is roughly 45% of the total effect, with the rest likely coming from 23.5 and 23.0 or their interaction. I also notice gen_8's original output matches gen_7 exactly, so the outputs may have been cached.

I'm going to check whether the outputs are cached, since the original in gen_8 matched gen_7 exactly, even though separate passes shouldn't reproduce each other like that. Next I'll run the 23.5 and 23.0 pair on its own, then decide whether to move on to a dose-response test on A.

