I'm looking at why the model runs so long: even the sin 0° case took about 7,000 tokens before closing. My working guess is that it says "Done thinking." and then fails to emit the closing tag when it should. One possibility I'm considering is that the closing token is being suppressed, and I'd test that by teacher-forcing texts where it says "Done thinking."

I'm weighing a second explanation: the model may be stuck in repetition loops, copying the same one or two lines over and over with no way out. I'd measure where each loop starts and how long its period is, which would show whether the failure begins early or only after a long ramble.

