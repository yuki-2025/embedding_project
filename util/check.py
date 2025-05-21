import os

def get_size(path):
    """Return total size of the file or folder at given path."""
    total_size = 0
    if os.path.isfile(path):
        total_size = os.path.getsize(path)
    elif os.path.isdir(path):
        # Walk the directory and sum the sizes of all files
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except Exception as e:
                    print(f"Error accessing {fp}: {e}")
    return total_size

def human_readable_size(size, decimal_places=2):
    """Convert size in bytes into a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024
    return f"{size:.{decimal_places}f} PB"

def main():
    path = input("Enter the path: ").strip()
    
    if not os.path.exists(path):
        print("The path does not exist. Please provide a valid path.")
        return
    
    # List only the immediate items in the provided directory
    try:
        items = os.listdir(path)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    if not items:
        print("The directory is empty.")
        return

    print(f"Sizes of items in '{path}':")
    for item in items:
        item_path = os.path.join(path, item)
        size = get_size(item_path)
        print(f"{item}: {human_readable_size(size)}")

if __name__ == "__main__":
    main()