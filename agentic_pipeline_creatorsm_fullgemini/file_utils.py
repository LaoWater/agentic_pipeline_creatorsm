# file_utils.py
import os
import re


def get_filename_base(subject: str, num_words: int = 4) -> str:
    """
    Generates a filename base from the first few words of the subject.
    """
    words = re.findall(r'\w+', subject.lower())
    return "_".join(words[:num_words])


def ensure_platform_folder_exists(base_output_folder: str, platform_name: str) -> str:
    """
    Ensures that the output folder for a specific platform exists and returns its path.
    Example: <base_output_folder>/facebook/
    """
    platform_folder_path = os.path.join(base_output_folder, platform_name)
    os.makedirs(platform_folder_path, exist_ok=True)
    return platform_folder_path


def save_text_content(directory_path: str, filename_base: str, text_content: str) -> str:
    """
    Saves the text content to a .txt file in the specified directory.
    Returns the full path to the saved text file.
    """
    text_file_path = os.path.join(directory_path, f"{filename_base}.txt")
    try:
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"📝 Text content saved to: {text_file_path}")
        return text_file_path
    except IOError as e:
        print(f"🚨 Error saving text file {text_file_path}: {e}")
        raise
