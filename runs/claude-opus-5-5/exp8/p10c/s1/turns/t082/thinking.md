I'm checking the record's structure: it has 'id', 'source', 'answer', and 'samples', and 'samples' is a list of sample dicts. Next I want to see the keys of those sample dicts and the signatures of the helper functions, possibly their source too.

I'm now looking at the helper functions, planning to print their signatures and the sample dict keys first. If the source is available, I'll also pull the first 1500 characters of `analyze` and `loop_info` so the output stays within the limit.

