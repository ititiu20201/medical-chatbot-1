import os

def create_project_structure():
    """Create the initial project directory structure."""
    
    # Define the directory structure
    directories = [
        'data/raw',
        'data/processed',
        'data/models',
        'src/data',
        'src/models',
        'src/utils',
        'src/training',
        'src/inference',
        'tests',
        'notebooks',
        'docs',
        'configs'
    ]
    
    # Create directories
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        # Create __init__.py files in Python package directories
        if dir_path.startswith('src/'):
            init_file = os.path.join(dir_path, '__init__.py')
            if not os.path.exists(init_file):
                open(init_file, 'a').close()
        # Create .gitkeep files in empty directories
        else:
            gitkeep_file = os.path.join(dir_path, '.gitkeep')
            if not os.path.exists(gitkeep_file):
                open(gitkeep_file, 'a').close()

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")