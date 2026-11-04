I had submitted a placeholder forward block with just one problem rather than the full S1 list, which cost a generation. This time the forward block has to contain the complete set of 256 ids written out, so I'm going to print the S1 id list in the kernel first and then paste it in, since roughly 8k characters of JSON is what the output has to hold.

I'm thinking about the output limit, roughly 10k characters, so the 256-id JSON list will fit. I'll print it compactly with json.dumps before pasting it into the forward block, and I'm checking whether there's any way to reference the ids by file instead of writing them out literally.

