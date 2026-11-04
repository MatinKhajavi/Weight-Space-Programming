I'm checking the token counts, which look high for 4000 characters: about 1.6 characters per token, likely from LaTeX or heavy whitespace. The 651-token sample works out to about 6 characters per token, so I'm guessing there's repeated whitespace there too. Since I need at most 2048 tokens per text, I'm looking for windows of 1800 tokens or fewer, and precalculus_742 at 1975 is just under the limit.

I'm weighing whether to pick windows around intermediate_algebra_948, which has a high token-to-character ratio, and trim the window to around 3000 characters to keep it under the token limit. I'd also be typing less JSON-escaped text that way, which avoids errors from the LaTeX backslashes.

