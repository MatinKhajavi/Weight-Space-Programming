Total comes to 469k tokens, which fits under the 524k limit. Next is the TF baseline pass on the original model across the 106 texts, requesting top-20 logprobs, which works out to roughly 246 MB of arrays, so that's manageable. I'm also wondering whether to request hidden states alongside it.

Next up is a quick check on whether the forward tool actually takes raw strings and tokenizes them the way I expect, then I'll decide whether any hidden-state or MLP activation arrays are worth requesting. Padding makes those large, so I'm leaning against them. I also want to update the notes file before the baseline pass in case of a restart.

