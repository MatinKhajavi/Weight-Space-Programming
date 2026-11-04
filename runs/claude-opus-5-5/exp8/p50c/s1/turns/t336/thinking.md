I'm realizing the earlier large file-based TF runs for corpus1 and conc probably never went through, since every previous pass used just two inline texts. That suggests the file mechanism may not exist, or I missed it.

I'm weighing whether a file reference might work for the teacher-forced request, since the earlier passes only ever used two inline texts. A failed call doesn't count against anything, so it's worth one try, and if the error says texts or chats are required, that will settle that there's no file mechanism.

If the file-path key fails, I'll fall back to the two-text inline check, though short texts of roughly a thousand tokens each would show little. I'll try "texts_file" first and adjust if the error lists the accepted keys.

