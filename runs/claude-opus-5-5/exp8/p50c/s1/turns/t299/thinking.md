The notes write seems to have failed, probably because the string has "27.3% cut" in it, and "% c" reads as a format specifier. So the notes weren't saved, and the brake-target control edit wasn't created either.

The notes write probably failed because the "% c" in "27.3% cut" got read as a format specifier, so nothing was saved and the brake-target control edit wasn't made. The head restoration did run, so the L15/L23 o_proj heads are back to C and L22 down is still at C, meaning the edit matches C exactly. Next I'll redo the notes with an f-string and set up the control edit, and check that the restored C tensors match the /work/best_C_x2 safetensors files.

