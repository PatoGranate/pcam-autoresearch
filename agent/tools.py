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

# Repository absolute path
REPO_ROOT = Path(__file__).parent.parent.resolve()

def resolve_repo_path(path_str):
    """ Turn input path string into absolute Path

    Args:
        path_str (str) : input folder/file path string
    
    Returns:
        root_path (Path) : absolute path for input path string
    """
    path = Path(path_str)
    root_path = Path(REPO_ROOT / path).resolve()

    if root_path.is_relative_to(REPO_ROOT):
        return root_path
    else:
        raise PermissionError


def list_files(path_str):
    """ List the contents of an input folder

    Args:
        path_str (str) : input folder path string
    Returns:
        contents_str (str) : names of files and subdirectories within input folder path
    """
    folder_contents = []

    try:
        # Get absolute path
        root_path = resolve_repo_path(path_str)

        # For each item in the root path, add to list with / if folder
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

    # If no contents in folder return (empty folder), else return contents separated by new lines
    if len(folder_contents) == 0:
        contents_str = "(empty folder)"
    else:
        folder_contents.sort()
        contents_str = "\n".join(folder_contents)

    return contents_str


# Tool dictionary, name : function
TOOL_FUNCTIONS = {"list_files": list_files}


def run_tool(name, tool_input):
    if name in TOOL_FUNCTIONS:
        try: 
            return TOOL_FUNCTIONS[name](**tool_input)
        except TypeError:
            return f"Error: {name} tool received unexpected inputs"

    else:
        return f"Error: Unknown tool: {name}"


if __name__ == "__main__":
    output = run_tool("list_files", {"path_str": "x"})
    print(output)