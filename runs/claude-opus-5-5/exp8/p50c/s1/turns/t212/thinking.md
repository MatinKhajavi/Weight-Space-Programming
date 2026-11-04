I've got attention data for fwd_18 across all the FA layers on the two C-loop windows, plus the hidden and MLP activations, and fwd_11 has its edit-A windows. Next I'm setting up a per-head "line-start retrieval" score: for query positions where the token ends in a newline, I'll measure how much attention goes to key positions that hold line-initial tokens.

I'm working out the head-level scoring now, checking how much attention the line-start query positions put on earlier line-initial tokens, then comparing that against induction and previous-token mass across the chosen layers.

I'm defining a per-head score: for newline-ending query positions, how much attention lands on earlier lines' first content tokens. Since the whitespace token before a line's first word matches the earlier occurrence, that retrieval is essentially the same signal as induction.

I'm now defining three per-head metrics for the C windows: overall induction mass, line-start mass on earlier first-content tokens, and the portion of that line-start mass not explained by a plain previous-token match.

