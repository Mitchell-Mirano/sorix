import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Clean up the mess from the previous run (the \' issue)
    content = content.replace("\\'BCELossWithLogits'", "'BCELossWithLogits'")
    content = content.replace("\\'CrossEntropyLoss'", "'CrossEntropyLoss'")
    content = content.replace("\\'ReLU'", "'ReLU'")
    content = content.replace("\\'Sigmoid'", "'Sigmoid'")
    content = content.replace("\\'Tanh'", "'Tanh'")
    content = content.replace("\\'get[", "'get[")
    
    # Generic cleanup for any \' before a literal string in Tensor calls
    import re
    content = re.sub(r'Tensor\(([^,]+),\s*([^,]+),\s*\\\'', r"Tensor(\1, \2, '", content)

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
