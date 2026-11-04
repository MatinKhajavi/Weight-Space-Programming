I'm restoring v4 exactly, then recording the pass 9 result in the notes. Next I'll build a fresh confirmation set for pass 10, leaving out every id already used in the diagnostic, training, and generation files.

I'm weighing a 128/128 split between MATH-500 and GSM8K against the earlier 192/64 mix, since both benchmarks seem to matter for the confirmation. I'm also checking the v4-versus-original gap on GSM8K from the existing diagnostic data, to see which mix gives enough power on each subset.

I'm settling on 128 MATH and 128 GSM8K, since both benchmarks need coverage and the expected gap is large enough that either split is clearly significant. For the MATH half I'll match the level distribution, which works out to 11, 23, 27, 33, and 34 items across the five levels. I'll sample within levels and let the subject mix follow naturally. The seed will be 777, and the pass will be a single "both" generate request with 256 problem ids.

I'm settling the problem list for the pass: 256 ids, with the MATH half split across the five levels as 11, 23, 27, 33, and 34 items. I'll write the ids out compactly so they fit in the output and can go straight into the forward block.

Still writing the id list for the 256-item confirmation set...

I'm computing the v4-versus-original gap separately for the setA and diagS sources, then updating the notes and building the fresh confirmation set with the 256 ids.

