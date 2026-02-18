import os
import re

def fix_file(filepath):
    # Skip binary or non-python files
    if not filepath.endswith('.py'):
        return
        
    with open(filepath, 'r') as f:
        content = f.read()

    # Change "from sorix import tensor" to "from sorix import Tensor, tensor" 
    # if it doesn't already have Tensor
    if 'from sorix import' in content or 'from sorix.tensor import' in content:
        content = re.sub(r'from sorix(\.tensor)? import ([^T\n]*)\btensor\b', r'from sorix\1 import \2Tensor, tensor', content)

    # Fix isinstance(x, tensor) -> isinstance(x, Tensor)
    content = re.sub(r'isinstance\s*\(([^,]+),\s*\btensor\b', r'isinstance(\1, Tensor', content)

    # Fix type hints in tests if any
    content = re.sub(r':\s*\btensor\b', ': Tensor', content)
    content = re.sub(r'->\s*\btensor\b', '-> Tensor', content)

    with open(filepath, 'w') as f:
        f.write(content)

def main():
    test_dir = '/home/mitchellmirano/Desktop/MitchellProjects/sorix/tests'
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            fix_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
