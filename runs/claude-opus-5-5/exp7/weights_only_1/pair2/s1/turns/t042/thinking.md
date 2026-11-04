I'm checking whether any attention head's share of positional attention shifts significantly, and none does: every |z| stays at or under about 2.9, which is within what noise would produce across 72 heads. The late copy heads show nothing either, so the behavioural difference doesn't seem to come from changes in relative-position attention.

The positional check turned up nothing: the copy heads show no shift either, so the difference doesn't seem to lie in where they attend. It probably sits in what the heads transmit or in how the answer gets written late on. This doesn't change my view, so I'll give the final answer and mention it.

