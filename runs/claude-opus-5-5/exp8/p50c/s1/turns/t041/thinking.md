I'm reorienting since this is a restarted conversation, and the notes are sparse with an empty pass log. I'm checking which files exist and what kernel variables are available. I've apparently used two forward passes (gen_3 and fwd_4), and I'm trying to work out whether the numbering started at 3 or whether passes 1 and 2 were something else.

The per-layer table points to full-attention layers, which seems to line up with the annotations at layers 4, 8, 12, 16, 20 and 24. The "pre / loop / diff" column is probably measuring how much a direction shifts the logits for some token before and during the loop, and layer 16 stands out with a diff of 2.49.

I'm checking the kernel state first: which variables exist, what the notes say, and what the gen_3 summary showed. I'll print the notes file and summarize gen_3.jsonl by accuracy and failure category. The Python budget and forward-pass count both have plenty of room left.

