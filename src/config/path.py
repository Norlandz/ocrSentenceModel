import os
from pathlib import Path


def get_project_root():
    """Returns the absolute path to the parent of the `src` directory, which is the project root."""
    # return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pathProjRoot = Path(__file__).resolve().parent.parent.parent
    print(">> get_project_root: " + pathProjRoot.as_posix())
    return pathProjRoot
