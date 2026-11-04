The forward pass hit a server-side engine initialization failure, so it doesn't count. I'm weighing whether to retry the same request or drop the activations to cut down on memory risk, though the error looks like it comes from engine startup rather than from the activations themselves. I'll retry once, and if it fails again, I'll change course.

I'm dropping the activations so the request is smaller, and I'll give the explanation before the code block.

