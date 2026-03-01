#!/usr/bin/env python3
"""
Script to update Colab links in notebooks to match the current git branch.
This ensures that when documentation is built from different branches,
the Colab links point to the correct branch.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def get_git_branch():
    """Get the current git branch name."""
    # First, try to get from environment variable (set by CI/CD)
    branch = os.environ.get('GIT_BRANCH')
    
    if branch:
        return branch
    
    # If not in CI, try to detect from git
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to 'develop' if git is not available
        return 'develop'


def update_notebook_links(notebook_path: Path, target_branch: str):
    """Update Colab links in a notebook to point to the target branch."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        modified = False
        
        # Define replacement patterns
        github_base = 'https://github.com/Mitchell-Mirano/sorix'
        github_blob = f'{github_base}/blob/{target_branch}'
        colab_base = f'https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/{target_branch}'
        
        # Documentation base URL - use GitHub Pages in CI, localhost in local dev
        is_ci = os.environ.get('CI', '').lower() == 'true'
        if is_ci:
            docs_base = 'https://mitchell-mirano.github.io/sorix'
        else:
            docs_base = 'http://127.0.0.1:8000/sorix'
        
        # Process all cells
        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type')
            if cell_type in ['markdown', 'code']:
                source = cell.get('source', [])
                if isinstance(source, list):
                    new_source = []
                    for line in source:
                        new_line = line
                        
                        # Update pip install command (usually in code cells)
                        if 'pip install' in new_line and 'sorix @ git+' in new_line:
                            import re
                            replaced_line = re.sub(
                                r'(github\.com/Mitchell-Mirano/sorix\.git@)[^/\'"]+',
                                rf'\1{target_branch}',
                                new_line
                            )
                            if replaced_line != new_line:
                                new_line = replaced_line
                                modified = True

                        # Replace Jinja2 macros with actual values
                        if '{{ github_blob }}' in new_line:
                            new_line = new_line.replace('{{ github_blob }}', github_blob)
                            modified = True
                        
                        if '{{ colab_base }}' in new_line:
                            new_line = new_line.replace('{{ colab_base }}', colab_base)
                            modified = True
                        
                        if '{{ github_base }}' in new_line:
                            new_line = new_line.replace('{{ github_base }}', github_base)
                            modified = True
                        
                        if '{{ branch }}' in new_line:
                            new_line = new_line.replace('{{ branch }}', target_branch)
                            modified = True
                        
                        if '{{ docs_base }}' in new_line:
                            new_line = new_line.replace('{{ docs_base }}', docs_base)
                            modified = True
                        
                        # Also replace hardcoded Colab links (for backward compatibility)
                        if cell_type == 'markdown' and 'colab.research.google.com/github/Mitchell-Mirano/sorix/blob/' in new_line:
                            import re
                            replaced_line = re.sub(
                                r'(colab\.research\.google\.com/github/Mitchell-Mirano/sorix/blob/)[^/]+/',
                                rf'\1{target_branch}/',
                                new_line
                            )
                            if replaced_line != new_line:
                                new_line = replaced_line
                                modified = True
                        
                        # Replace hardcoded GitHub blob links
                        if cell_type == 'markdown' and 'github.com/Mitchell-Mirano/sorix/blob/' in new_line and 'colab' not in new_line:
                            import re
                            replaced_line = re.sub(
                                r'(github\.com/Mitchell-Mirano/sorix/blob/)[^/]+/',
                                rf'\1{target_branch}/',
                                new_line
                            )
                            if replaced_line != new_line:
                                new_line = replaced_line
                                modified = True
                        
                        new_source.append(new_line)
                    
                    cell['source'] = new_source
        
        # Write back if modified
        if modified:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            try:
                rel_path = notebook_path.relative_to(Path.cwd())
            except ValueError:
                rel_path = notebook_path
            print(f"✓ Updated {rel_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error processing {notebook_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main function to update all notebooks."""
    # Get the target branch
    branch = get_git_branch()
    print(f"Updating notebook links to branch: {branch}")
    
    # Find all notebooks in docs directory
    docs_dir = Path('docs')
    if not docs_dir.exists():
        print(f"Error: docs directory not found", file=sys.stderr)
        sys.exit(1)
    
    notebooks = list(docs_dir.rglob('*.ipynb'))
    
    if not notebooks:
        print("No notebooks found")
        return
    
    print(f"Found {len(notebooks)} notebook(s)")
    
    updated_count = 0
    for notebook in notebooks:
        if update_notebook_links(notebook, branch):
            updated_count += 1
    
    print(f"\nUpdated {updated_count}/{len(notebooks)} notebook(s)")


if __name__ == '__main__':
    main()
