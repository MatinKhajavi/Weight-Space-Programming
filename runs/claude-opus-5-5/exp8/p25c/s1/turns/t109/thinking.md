I'm checking what pid2prob holds: it looks like it maps each id to the problem text string rather than a dict, so it doesn't include answers. To get the answers I'll need to look at the gen record's answer field, or the id2rec dictionary with about 15,000 entries.

I'm going with id2rec, which has about 15,000 entries and seems to map each id to a record that includes the answer.

