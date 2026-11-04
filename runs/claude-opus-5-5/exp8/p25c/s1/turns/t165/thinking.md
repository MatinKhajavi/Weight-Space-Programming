I'm going to look at the source of a few helper functions (find_loop, loop_info, categorize, cat_of, near_rep, paired, pooled) so I can reuse them, using inspect.getsource. The catch is that functions defined through exec in the kernel may not have retrievable source, so I'll just try it and see.

