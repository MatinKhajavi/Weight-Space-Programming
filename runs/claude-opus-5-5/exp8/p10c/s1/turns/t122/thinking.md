I'm checking the sample records first: each one holds a list of dicts with the text and some extra fields, so I want to see the exact keys in the first sample. After that I'll run the paired comparison of v7 against v4, then restore v4 from the saved best_4 files by loading each one and writing it back with the save function.

I'm verifying that the restored edits match what's expected relative to the original weights, such as the L11 o_proj columns 256:512 being scaled to half their original values.

