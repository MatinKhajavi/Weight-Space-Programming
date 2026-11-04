I'm going to try a forward block that points at a texts file, since there's no documentation to go on. If that fails it won't count, though I'm uneasy the tool might quietly handle the path differently. I'm also wondering whether it would accept the texts field as a plain path string rather than a list.

I'm weighing whether the tool might accept the texts as a file path, since the deep_texts.json file in /work hints I may have used one before. A plain list of strings seems more likely to be what it expects, so I'll save a list file and try a "texts_file" key. That key is the safer choice, because an unsupported key should just fail without counting.

