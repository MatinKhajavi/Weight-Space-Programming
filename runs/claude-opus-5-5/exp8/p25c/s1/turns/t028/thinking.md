I'm checking the token estimates against the character counts, and they roughly line up once the 120-token lead, the six periods, and the prompt are added. The total comes to about 7.7k characters, so I'm considering adding more texts, making it 14 at around 13.5k characters. Hidden states across all 25 layers plus the MLP outputs for all 24 layers would come to about 5.5k rows.

I'm settling on 14 texts of about 1,000 characters each, which would bring the total to roughly 14k characters. That's still fine in terms of typing cost. I'm also weighing whether to add two control texts, like a GSM8K problem with a reference solution, but I'll leave that out for now. Next I'll build the 14 texts and print them as a JSON list in chunks under 10k characters so I can copy them.

