I'm noticing that even a simple case like "Compute sin 0°" runs past 2048 tokens of thinking, with the model second-guessing itself ("Wait, what if the user meant radians?"). That looks like heavy overthinking, and with a 32768 token budget it may often never finish. Next I want to look at the full sin 0 example to see how many of those "Wait" moments there are and how they're structured.

