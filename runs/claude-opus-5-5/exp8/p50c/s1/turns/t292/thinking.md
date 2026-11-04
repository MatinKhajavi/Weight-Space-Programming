I'm weighing how to pass the 512 IDs: printing them compactly and expanding by hand risks typos, so I'm considering printing the full list in two halves instead. I'm also thinking about trimming the test to a smaller set, starting with S1.

I'm weighing a smaller 36-ID subset against the full 512, since the paired power looks decent but the problem pool is narrow. I'm also considering printing the 512 IDs in compact grouped form, though expanding them by hand could introduce errors.

I'm still deciding how to hand over the 512 IDs, since typing out the full list by hand invites transcription errors. Splitting it into two runs of about 256 each seems more manageable, though it means more Python calls.

I'm going to print the S1 and H1 IDs in compact grouped form in one run, then expand them into full IDs for the forward block. That keeps it to a single call, since the grouped output should come in around 2.6k characters.

Next I'll print the grouped IDs for both sets so I can write out the full list accurately, since typing them by hand is error-prone and any mistake would make the request fail.

