I'm weighing whether to run a positions pass first, since the trunk changed a lot with the VX setup and a bug there could spoil a full games run. A games run would expose broken performance too, though, and both cost the same single pass. Running positions with activations on the 36 test positions would give exact diagnostics, but I'm also considering just running games and debugging only if the results look bad, and the subtle bugs are the part I'm still uncertain about.

I'm going with a games run now, strong at 96 games and weak at 32, rather than a positions pass first. The concern is that subtle bugs, like a slightly wrong AX, might not show up in game results.

