Edit B pushed accuracy to 20.7%, still above the original 12.1%, but less good than Edit A's +22.7. It cut about as many heads as A did, yet produced far more wrong answers (42 against 9). So removing those extra four heads hurt the reasoning without reducing the loops further, meaning the loops aren't driven only by these heads, which are needed for correct computation.

Edit A still looks like the best candidate, so I want to see how much of its gain comes from each head. Testing A with only the 15.7 head ablated, or with 23.5 and 23.0 together, would show which ones matter. I could also try half strength, or a negative scale that flips the head's effect, though that might be risky. The bigger open question is why loops still show up in about 61% of cases with A, so I'll look at what those remaining loops look like and where they begin.

I'm checking whether the remaining loops in edit A come from some other mechanism, like the model being stuck on problems it can't solve, with the copying heads just making the repetition exact. To test that, I'd compare the edit A outputs against the original on the same items, looking at output length and difficulty, especially for items the original cut and edit A got right.

I'm checking whether the remaining loops are simply the hard MATH problems, which the model can't solve and keeps cycling on, by comparing the level distribution. Then I want to test A at half strength and each head's contribution separately, since a handful of passes should be enough to see the dose-response.

Next I want to see whether the remaining loops are carried by other heads, so I'll look at the edit-A cut samples and run a forward pass with attention over one of those loop windows. I can only print ASCII from the kernel, so any non-ASCII characters will show as '?'.

Next I want to see how the remaining loops break down by difficulty level, since the gen_6 and gen_7 runs are the ones to compare.

