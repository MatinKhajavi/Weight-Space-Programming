Some of the per-head QK changes are large, up to about 12 in magnitude, with mixed signs, so the diagonal of those matrices shifted noticeably. But I want to check the remaining heads before drawing conclusions from that.

I'm worried the rank-1 part of the QK change could be inflating these diagonal values. If the query-side and key-side directions line up, the trace picks up a constant "sink" component rather than real token matching. The massive dimensions, like 163 and 400, are the likely culprit, so I want to project out the top shared direction and recompute.

I'm checking whether the trace is just picking up the shared massive dimensions rather than token matching, so I'm considering excluding the top answer-residual dims (like 163 and 400) and recomputing the permutation null on the remaining ones. I'm also reading the signs: some heads decrease same-token matching, others increase it, so the pattern is mixed.

