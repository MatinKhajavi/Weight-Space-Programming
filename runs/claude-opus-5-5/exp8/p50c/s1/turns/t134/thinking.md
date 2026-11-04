I'm setting up the window selection next: the onsets need to be at least 3000 characters in, and the P values between 30 and 500. I still need to confirm how the tokenizer's encode call behaves before tokenizing the full thinking text for each candidate.

I'm picking the window size and count now: about 800 tokens after the onset should cover at least eight loop cycles. I also want to start each window at a line break, and I'm working out how many windows fit within the memory budget, settling on ten.

