Loops show up around 18k characters, roughly 20% into the text. Of the 42 that finished, 41 were correct, so the model handles the math well when it actually gets to the end. The real problem is that it often never finishes. I'm also wondering whether the literal "<think>" tag appears inside the text, since the thinking begins right after that tag in the prompt.

I'm checking whether the model's looping is just its normal behavior or something deliberately broken, and the task framing points toward repetition being the main failure to explain.

I'm looking at what kind of failure the looping is: the model has already reached the answer, yet it keeps going with "Wait" and then falls into a verbatim cycle. I'm weighing two mechanisms, excess self-checking and copying from earlier repeats, and I'm noting that fixing the repetition is in scope while forcing the model to skip its thinking is not. Next I want to examine the text around the loop onset to see how the transition happens.

