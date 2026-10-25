from pathlib import Path
import tempfile

tools = [
    {
        "name": "list_files",
        "description": "Returns a list of files and subfolders of an input folder. Input folder must be path with respect to repo root; "
        "returns files as strings, and folders contain a slash '/' at the end of the name to indicate a directory. If input path is "
        "invalid or outside the repository, returns error information and tool call will be rejected. Use '.' for root repository",
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
    },
    {
        "name": "write_file",
        "description": "Write to an existing file, or create a new one to write - replaces the whole file. The input to the tool must be "
        "a str relative to the repo root, and be within the repository. If the input path is invalid or outside the repository, returns "
        "error information and tool call will be rejected. Within the repository, access is only granted to write within configs/agent; "
        "anything else will return PermissionError. The file_content parameter should contain the full text of the file, and it replaces "
        "the WHOLE file. Use real line breaks, and for YAML use spaces for indentation, not tabs. Uses UTF-8 encoding unless otherwise "
        "specified by the 'encoding' argument",
        "input_schema": {
            "type": "object",
            "properties": {
                "path_str": {
                    "type": "string",
                    "description": "Path to input file relative to repo root, e.g. 'configs/agent/test.txt'"
                },
                "file_content": {
                    "type": "string",
                    "description": "String containing the content to be written within the input file string: path_str. e.g. 'Hello world'"
                },
                "encoding": {
                    "type": "string",
                    "description": "encoding name as a string, e.g. 'cp1252'. If none given, UTF-8 used as default."
                }
            },
            "required": ["path_str", "file_content"]
        }
    }
]


class ToolError(Exception):
    pass

# Repository absolute path
REPO_ROOT = Path(__file__).parent.parent.resolve()
# Limit access to files/folders under each specific action. list_files is least restrictive, write_files is most restrictive
ACTION_PERMISSIONS = {"list": ["configs/", "pcam/", "scripts/", "runs/", "agent/"], 
                      "read": ["configs/", "pcam/", "scripts/", "runs/", "agent/", "train.py", "detect_noise.py", "README.md", "CLAUDE.md",
                                "requirements.txt", ".gitignore", "notes.md"], 
                      "write": ["configs/agent/"]}


def resolve_repo_path(path_str, action):
    """ Turn input path string into absolute Path

    Args:
        path_str (str) : input folder/file path string
        action (str) : identity of tool calling the resolve_repo_path function
    
    Returns:
        root_path (Path) : absolute path for input path string
    """
    path = Path(path_str)
    root_path = Path(REPO_ROOT / path).resolve()

    if root_path.is_relative_to(REPO_ROOT):
        # Allow the agent to list everything one level deep, but prevent .venv, .vscode, .claude etc for context and token preservation
        if action == "list" and root_path == REPO_ROOT:
            return root_path

        # If the file/folder being accessed is within the permitted list for that action, resolve the path, otherwise don't.
        for permission_path_str in ACTION_PERMISSIONS[action]:
            if root_path.is_relative_to(Path(REPO_ROOT / permission_path_str).resolve()):
                return root_path
            
    raise PermissionError


def list_files(path_str):
    """ List the contents of an input folder

    Args:
        path_str (str) : input folder path string
    Returns:
        contents_str (str) : names of files and subdirectories within input folder path
    """
    if path_str == "":
        raise ToolError("Path string input was an empty string")
    
    folder_contents = []

    try:
        # Get absolute path
        root_path = resolve_repo_path(path_str, action="list")

        # For each item in the root path, add to list with / if folder
        for subdir in root_path.iterdir():
            if subdir.is_dir():
                folder_contents.append(f"{subdir.name}/")
            else:
                folder_contents.append(subdir.name)

    except PermissionError as e:
        raise ToolError(f"Path access denied (or there was an access problem: {e})") from None
    
    except NotADirectoryError as e:
        raise ToolError(f"Input path is not a folder: {e}") from None
    
    except FileNotFoundError as e:
        raise ToolError(f"Input path was not found: {e}") from None

    # These don't do anything, but serve as a reminder for run_tools to handle exceptions that weren't previously caught
    except Exception:
        raise

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

    Returns:
        file_content (str) : content read from the input file
    """
    if path_str == "":
        raise ToolError("Path string input was an empty string")
    
    # Set a maximum character limit for large files to be truncated early
    max_char = 10000

    try:
        root_path = resolve_repo_path(path_str, action="read")

        if not root_path.exists():
            raise ToolError("Input path was not found")

        if root_path.is_file():
            # Open the file in binary mode and search for b"\x00" in the first thousand characters to determine if file is binary
            with root_path.open(mode = "rb") as file:
                chunk = file.read(1000)
                if b"\x00" in chunk:
                    raise ToolError("File looks binary and can't be read as text")

            # Read file content
            with root_path.open(mode = "r", encoding = encoding) as file:
                file_content = file.read(max_char+1)

            # If over length limit, truncate and return as info
            if len(file_content) == 0:
                return "(empty file)"
            elif len(file_content) > max_char:
                file_length = root_path.stat().st_size
                return f"File was too long and had to be truncated at a max of {max_char} characters. File length was: {file_length} bytes. Read file content: {file_content[:max_char]}"

            else:
                return file_content
            
        elif root_path.is_dir():
            raise ToolError("Path is a folder, not a file. Possibly look at list_files tool instead?")

        else:
            raise ToolError("Path did not lead to a file")

    # If decoding error, return it 
    except UnicodeDecodeError as e:
        raise ToolError(f"Failed to decode file with {encoding}: {e}")
    
    except PermissionError as e:
        raise ToolError(f"Path access denied (or there was an access problem: {e})")

    except OSError as e:
        raise ToolError(f"Failed to read file: {e}")
    
    # If encoding input is incorrect return error
    except LookupError as e:
        raise ToolError(f"Entered encoding: {encoding} was not valid: {e}")

    # These don't do anything, but serve as a reminder for run_tools to handle exceptions that weren't previously caught
    except Exception:
        raise


def write_file(path_str, file_content, encoding="UTF-8"):
    """ Write file content to an input file given by path_str with a specific encoding, default UTF-8
    
    Args: 
        path_str (str) : input file path string
        file_content (str) : content to be written to the input file
        encoding (str) : string name for file encoding
    
    Returns:
        msg (str) : informational message of what the function achieved
    """
    temp_path = None
    try:
        root_path = resolve_repo_path(path_str, action="write")

        # Check for encoding errors before ever editing or creating the file to prevent halfway errors
        data = file_content.encode(encoding)

        # If input is a folder, write_text will not work, so raise a ToolError
        if root_path.is_dir():
            raise ToolError("input path_str was not a file, so cannot be written")

        # Default message is simply writing
        msg = f"Wrote input file content to {path_str}"

        if not root_path.exists():
            root_path.parent.mkdir(parents=True, exist_ok=True)
            # If a file is created, change the message to include this
            msg = f"Created {path_str}, and wrote input file content to it"

        if file_content == "":
            msg = f"input file_content was empty; {path_str} is now a blank file"

        # Write the content to a temporary file, and then replace the root_path with the new content to prevent midway failures
        with tempfile.NamedTemporaryFile(mode="wb", dir=root_path.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(data)
        temp_path.replace(root_path)

        return msg

    except PermissionError as e:
        raise ToolError(f"Path access denied (or there was an access problem: {e})")

    except UnicodeEncodeError as e:
        raise ToolError(f"Failed to encode file with {encoding}: {e}")

    # If encoding input is incorrect return error
    except LookupError as e:
        raise ToolError(f"Entered encoding: {encoding} was not valid: {e}")

    except OSError as e:
        raise ToolError(f"Failed to write file: {e}")

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# Tool dictionary, name : function
TOOL_FUNCTIONS = {"list_files": list_files, "read_file": read_file, "write_file": write_file}


def run_tool(name, tool_input):
    if name in TOOL_FUNCTIONS:
        try: 
            result = TOOL_FUNCTIONS[name](**tool_input)
            return result, False
        
        except ToolError as e:
            return str(e), True
        
        except TypeError as e:
            return f"Error: {name} tool received unexpected inputs: {e}", True

        except Exception as e:
            print(f"Unexpected error in {name}: {e}")
            return f"Unexpected error in {name}: {e}", True

    else:
        return f"Error: Unknown tool: {name}", True


if __name__ == "__main__":
    output = run_tool("write_file", {"path_str": "configs/agent/sub/deeptest.yaml", "file_content": "cafe", "encoding": "UTF-8"})
    print(output)