Level values come as strings like "Level 5", and some are "Level ?", so the parsing needs to handle that. The earlier error at the pool comprehension means the `used` set was already built, so I'm going to redo the sampling step with the level parsing included.

