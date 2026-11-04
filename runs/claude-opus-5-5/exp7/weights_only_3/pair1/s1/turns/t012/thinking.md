I'm checking whether any gain values are zero, since that would break a division. I'm considering guarding the division so it only happens where the gain is meaningfully non-zero, and wondering whether to skip dividing by the gain entirely, since that might distort the Adam updates.

