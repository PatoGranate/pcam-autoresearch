from pathlib import Path

tools = [
    {
        "name": "list_files",
        "description": "Returns a list of files and subfolders of an input folder. Input folder must be path with respect to repo root; "
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
    },
    {
        "name": "read_file",
        "description": "Read the contents of a specified input file. The input to the tool must be a str relative to the repo root, and "
        "be a file within the repository. If the input path is invalid or outside the repository, returns error information and tool "
        "call will be rejected. Uses UTF-8 decoding unless otherwise specified by the 'encoding' argument.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path_str": {
                    "type": "string",
                    "description": "Path to input file relative to repo root, e.g. 'agent/agent.py'"
                },
                "encoding": {
                    "type": "string",
                    "description": "encoding name as a string, e.g. 'cp1252'. If none given, UTF-8 used as default."
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


def read_file(path_str, encoding = "UTF-8"):
    """ Read the contents of a file
    
    Args:
        path_str (str) : input file path string
        encoding (str) : string name for file encoding
    """
    # Set a maximum character limit for large files to be truncated early
    max_char = 10000

    try:
        root_path = resolve_repo_path(path_str)

        if not root_path.exists():
            return "Error: Input path was not found"

        if root_path.is_file():
            # Open the file in binary mode and search for b"\x00" in the first thousand characters to determine if file is binary
            with root_path.open(mode = "rb") as file:
                chunk = file.read(1000)
                if b"\x00" in chunk:
                    return "Error: File looks binary and can't be read as text"

            file_length = root_path.stat().st_size

            # Compare file length to max car and determine whether to truncate or not
            if file_length < max_char:
                return root_path.read_text(encoding = encoding)
            else:
                with root_path.open(mode = "r", encoding = encoding) as file:
                    file_content = file.read(max_char)
                    return f"File was too long and had to be truncated at a max of {max_char} characters. File length was: {file_length}. Read file content: {file_content}"
        
        elif root_path.is_dir():
            return "Error: Path is a folder, not a file. Possibly look at list_files tool instead?"

        else:
            return "Error: Path did not lead to a file"

    # If decoding error, return it 
    except UnicodeDecodeError as e:
        return f"Error: Failed to decode file with {encoding}: {e}"
    
    except PermissionError as e:
        return f"Error: Paths pointing outside the repository may not be read, or there was an access problem: {e}"

    except OSError as e:
        return f"Error: Failed to read file: {e}"
    
    # If encoding input is incorrect return error
    except LookupError as e:
        return f"Error: Entered encoding: {encoding} was not valid: {e}"

    
# Tool dictionary, name : function
TOOL_FUNCTIONS = {"list_files": list_files, "read_file": read_file}


def run_tool(name, tool_input):
    if name in TOOL_FUNCTIONS:
        try: 
            return TOOL_FUNCTIONS[name](**tool_input)
        except TypeError:
            return f"Error: {name} tool received unexpected inputs"

    else:
        return f"Error: Unknown tool: {name}"


if __name__ == "__main__":
    output = run_tool("read_file", {"path_str": "README.md", "encoding": "cp1252"})
    print(output)