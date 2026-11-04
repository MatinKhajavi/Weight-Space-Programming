Line-initial "Maybe" fell from 2.68 to 1.71 in gen_21, while lowercase "maybe" went up from 0.54 to 0.84. That means gen_21 doesn't act as a brake on the maybe family through the 22.387 direction, since that would also suppress lowercase " maybe", which it doesn't, so the 22.387 doubling is still untested. Under C, line-initial "Maybe" sits at 2.68 per 1k against 0.46 in the original, roughly a sixfold increase.

I'm weighing the 22.387 doubling as the next target, since the brake neuron would suppress the Maybe family directly. I'm choosing between SH512 and H2 for the test: SH512 gives about 512 paired generations, roughly SE 2.0, and it's a fair fresh test for X6 because X6 wasn't selected on it. I'll run X6 on SH512 against C's outputs, then confirm on H2 or another fresh set if the result is positive.

I need the list of problem IDs for the forward request, but I don't have them in front of me, and printing all 512 would be too long for the output limit. I'll check whether the tool accepts a file path for the problem list instead, and if the first request fails I'll switch to reading the IDs in chunks.

I'm weighing whether to print the 512 problem IDs for the forward request, since listing them all would overrun the output limit. I could print them compactly, but that would take several Python runs, and H2 at 256 IDs would be a lighter option.

I'm weighing whether to pass the 512 problem IDs through as a file path or list them out, since printing them all would overrun the output limit. Another option is 128 problems at n=4, though that would need C's results for those same samples to compare against.

I'm checking whether the 36 discordant problems would make a cleaner comparison than SH512, since they already have C results on matched seeds at 14 samples each. Their pooled accuracy is around 0.74, and the ID list is short enough to paste directly.

I'm weighing whether to test on the 36 discordant problems or on a larger set. Those 36 give paired comparisons over 504 samples, but they only cover 36 problems. S1 alone would be 256 items, and the ID list would need printing in one go to get the format right.

I'm weighing whether to test the 22.387 doubling on the 36 discordant problems, which have matched C results at 14 samples each, or on a larger set like SH512. The 36 problems are cheap to list and give paired comparisons over 504 samples, though they cover fewer problems.

I'm weighing the 36 discordant problems as a smaller check, but they're a narrow sample, so a positive result there would still need a broader confirmation run. Running X6 on H1 and S1 together as one 512-item pass would be stronger, though I'd need to print those problem IDs first, possibly in grouped chunks to keep the output within limits.

I'm weighing how to pass the 512 problem IDs to the forward request, since typing them all out would be long and error-prone. I'll test whether the tool accepts a file path for the problem list, and if that fails I'll fall back to printing and pasting the IDs.

I'll try passing the problem list as a file path first, since that avoids pasting 512 IDs. If the request is rejected, it shouldn't use up a pass, and the error may show the format it expects. If that fails, I'll print the IDs and paste them in.

