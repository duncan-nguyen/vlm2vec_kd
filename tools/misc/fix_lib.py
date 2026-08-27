# Run directly from the repo root, e.g. `torchrun tools/misc/fix_lib.py`: put the repo root on
# sys.path so `import src.…` resolves without installing the project.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os

file_path = "./vlm/lib/python3.11/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py"

start_line = 140
end_line = 143

with open(file_path, "r") as f:
    lines = f.readlines()

with open(file_path, "w") as f:
    for i, line in enumerate(lines, start=1):
        if start_line <= i <= end_line:
            if not line.lstrip().startswith("#"):  # tránh comment lại 2 lần
                f.write("# " + line)
            else:
                f.write(line)
        else:
            f.write(line)

print(f"✅ Done! Lines {start_line}-{end_line} have been commented.")