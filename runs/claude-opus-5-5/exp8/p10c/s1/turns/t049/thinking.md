I'm going to compute the statistics next. For the finished samples, that means the count of "Wait" per 1k tokens, plus where </think> appears and how many "Wait"s there are. For the cut samples, I need to locate the loop onset, which I'll do by tokenizing the texts since there are about 14M tokens to process.

I'm going to look at whether the first 3000 tokens differ in "Wait" density between finished and looping samples, to see if heavy reflection comes before a loop. I also want the share of paragraphs that begin with "Wait", counting after blank lines or bullets, and then I'll tokenize all the texts in one batch.

