from pathlib import Path

tools = [
    {
        "name": "list_files",
        "description": "Returns a list of files and subfolders of an input folder. Input folder must be path with respect to repo root;"
        "returns files as strings, and folders contain a slash '/' at the end of the name to indicate a directory. If input path is "
        "invalid or outside the repository, returns error information and tool call will be rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path_str": {
                    "type": "string",
                    "description": "Path to input folder relative to repo root, e.g. 'configs/detection'"
                }
            },
            "required": ["path_str"]
        }
    }
]

REPO_ROOT = Path(__file__).parent.parent.resolve()


def resolve_repo_path(path_str):
    path = Path(path_str)
    root_path = Path(REPO_ROOT / path).resolve()

    if root_path.is_relative_to(REPO_ROOT):
        return root_path
    else:
        raise PermissionError


def list_files(path_str):
    folder_contents = []

    try:
        root_path = resolve_repo_path(path_str)

        for subdir in root_path.iterdir():
            if subdir.is_dir():
                folder_contents.append(f"{subdir.name}/")
            else:
                folder_contents.append(subdir.name)

    except PermissionError:
        return "Error: Paths pointing outside the repository may not be searched"
    
    except NotADirectoryError:
        return "Error: Input path is not a folder"
    
    except FileNotFoundError:
        return "Error: Input path was not found"

    if len(folder_contents) == 0:
        contents_str = "(empty folder)"
    else:
        folder_contents.sort()
        contents_str = "\n".join(folder_contents)

    return contents_str

if __name__ == "__main__":
    subdir = list_files("C:\\Windows")
    print(subdir)