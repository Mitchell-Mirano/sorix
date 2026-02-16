"""
Macros module for MkDocs to provide dynamic variables.
"""
import os
import subprocess


def define_env(env):
    """
    Define custom variables and macros for MkDocs.
    
    This function is called by mkdocs-macros-plugin.
    """
    
    # Detect the current git branch
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
    
    # Set the branch variable
    env.variables['branch'] = get_git_branch()
    
    # Also provide the full GitHub URL base for convenience
    env.variables['github_base'] = 'https://github.com/Mitchell-Mirano/sorix'
    env.variables['github_blob'] = f"{env.variables['github_base']}/blob/{env.variables['branch']}"
    
    # Colab base URL
    env.variables['colab_base'] = f"https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/{env.variables['branch']}"
    
    # Documentation base URL - use GitHub Pages in CI, localhost in local dev
    # Check if we're in CI/CD (GitHub Actions sets CI=true)
    is_ci = os.environ.get('CI', '').lower() == 'true'
    if is_ci:
        env.variables['docs_base'] = 'https://mitchell-mirano.github.io/sorix'
    else:
        env.variables['docs_base'] = 'http://127.0.0.1:8000'
