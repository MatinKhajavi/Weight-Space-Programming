I'm looking for another hypothesis to test that hasn't been checked yet. The key question is whether the shift from identity to related copying happens for every word, not just answer tokens. So I want to see whether late attention copies random words less in absolute terms, which would make the earlier controlled result for answer tokens look like an even larger drop. I'm weighing how to set up a proper null for that comparison.

I'm considering whether to split the late-head effect into output-matrix and value-projection contributions, comparing identity and related mappings to see where the difference actually lives.

