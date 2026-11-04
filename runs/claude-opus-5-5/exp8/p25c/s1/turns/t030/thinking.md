I'm composing the forward pass, which should return token logprobs, the top 20 per position, all 25 hidden-state layers, and all 24 MLP activation layers. At roughly 6.1k slots of fp32 that comes to about 2.7GB, which fits.

I'm now checking the JSON transcription, making sure the escapes and newlines are reproduced exactly as printed.

