I'm checking the shapes for the layer's output-times-value product: the attention output is 640x1024 and the value matrix is 256x640, so I'll tile the value matrix across the four heads to get a 1024x640 matrix that lines up.

