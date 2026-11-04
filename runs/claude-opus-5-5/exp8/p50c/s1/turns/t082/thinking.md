I'm checking the two windows, which are raw text rather than chat, with lengths of 1841 and 1500. The token counts don't line up with those lengths, so I suspect the text was cut to a character window and then retokenized. The onset setting for the second window also lands at its very end, which looks wrong.

I'm now suspecting the token counts disagree with the text lengths because the server may tokenize differently or the window was re-tokenized, so I want to recompute the tokens myself and check where the loop onset falls. I'll look at the token IDs directly and scan for the repeating period, then check how the helper functions find the loop.

