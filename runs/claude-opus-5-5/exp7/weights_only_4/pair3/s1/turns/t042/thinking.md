I've spotted the bug: the output projection is applied to the wrong intermediate shape, so I need to sum over the heads, applying each head's slice of the output weights to its value vectors instead.

