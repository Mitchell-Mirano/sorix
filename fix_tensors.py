import os
import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Fix internal tensor calls that use _children and _op
    # These usually look like tensor(data, (child1, child2), 'op', ...)
    # or tensor(data, (X,), 'ReLU', ...)
    # Regex designed to catch calls with at least 3 positional arguments or specific op strings
    content = re.sub(r'\btensor\s*\(([^,]+),\s*\(([^)]*)\),\s*\'', r'Tensor(\1, (\2), \'', content)
    content = re.sub(r'\btensor\s*\(([^,]+),\s*\[([^\]]*)\]\s*,\s*\'', r'Tensor(\1, [\2], \'', content)
    
    # Also catch calls where 2nd arg is a variable starting with _ (often _children)
    content = re.sub(r'\btensor\s*\(([^,]+),\s*(_[a-zA-Z0-9_]+),\s*\'', r'Tensor(\1, \2, \'', content)

    # 2. Fix type hints and return value annotations
    # Match ": tensor" (optionally with spaces)
    content = re.sub(r':\s*\btensor\b', ': Tensor', content)
    # Match "-> tensor"
    content = re.sub(r'->\s*\btensor\b', '-> Tensor', content)
    # Match "| tensor" and "tensor |"
    content = re.sub(r'\|\s*\btensor\b', '| Tensor', content)
    content = re.sub(r'\btensor\s*\|', 'Tensor |', content)
    # Match "[tensor]" (often in lists or Union)
    content = re.sub(r'\[\s*\btensor\s*\]', '[Tensor]', content)

    # 3. Fix isinstance calls
    content = re.sub(r'isinstance\s*\(([^,]+),\s*\btensor\b', r'isinstance(\1, Tensor', content)
    # Also handle tuples in isinstance: isinstance(X, (tensor, np.ndarray))
    content = re.sub(r'isinstance\s*\(([^,]+),\s*\(([^)]*)\btensor\b([^)]*)\)', 
                     lambda m: f"isinstance({m.group(1)}, ({m.group(2)}Tensor{m.group(3)}))", 
                     content)

    with open(filepath, 'w') as f:
        f.write(content)

def main():
    base_dir = '/home/mitchellmirano/Desktop/MitchellProjects/sorix/sorix'
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                fix_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
