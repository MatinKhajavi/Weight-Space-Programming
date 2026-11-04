The edit file checks out, so the failure has to be on the server side. I'm going to retry. My guess is that "both" mode needs two engines at once, and with max_tokens at 32768 and 256 sequences that could run out of GPU memory. I'll try "both" again without activations. If it still fails, running "edited" alone is an option, though that would rule out pairing.

