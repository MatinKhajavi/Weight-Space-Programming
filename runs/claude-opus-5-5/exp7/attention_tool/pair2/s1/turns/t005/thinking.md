I'm setting up the analysis: for each text I need to mark the guess tokens, the "I think ... is" statement region, and the question span. Then, at the last query position and the model-turn tokens, I'll measure attention mass on the guess tokens per layer and head for A and B, average the A-minus-B difference across texts, and pick out the top heads.

Now I'm pinning down the exact token indices for each text's guess span, noting the single-token and two-token cases separately. I'll check the span boundaries against the segment around the answer and then compare attention to the question span as a baseline.

