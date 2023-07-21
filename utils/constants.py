# %%
import os
import matplotlib.colors as mcolors

from pathlib import Path

# ANSI escape sequences of different colors for text output
reset = "\033[0m"
bold = "\033[1m"

# Colors
red = "\033[0;91m"
green = "\033[0;92m"
yellow = "\033[0;93m"
blue = "\033[0;94m"
magenta = "\033[0;38;5;199m"
cyan = "\033[0;96m"
white = "\033[0;97m"

# Bold colors
bgray = "\033[1;90m"
bred = "\033[1;91m"
bgreen = "\033[1;92m"
bblue = "\033[1;94m"
bmag = "\033[1;38;5;199m"
bcyan = "\033[1;96m"

# Colors palettes
CYBERPUNK_PALETTE = {
    "pink": "#ff6090",
    "yellow": "#f7d038",
    "cyan": "#00ffff",
    "orange": "#ffa270", 
    "green": "#7cff8c",
    "purple": "#9b4dff",
    "indigo": "#5c5cff",
    "teal": "#00c8a9",
}

HOPIUM_PALETTE = {
    "hopium":"#00e6b8",
    "red":"#ff6e68",
    "blue":"#6798c6",
    "orange":"#ffa23a",
    "purple":"#b36dbf",
    "green":"#2eb872",
    "pink": "#ff8ab8",
    "yellow": "#ffe94a",
}

### COLORMAP HOPIUM ###

# Define the colors of the gradient
COLORS = [
    "#e0332b",
    "#eb7d78",
    "#d4d4d7",
    "#09ffc7",
    "#00c699"
]

LINEWIDTH = 3

# Create a custom colormap using the defined colors
HOPIUM_CMAP = mcolors.LinearSegmentedColormap.from_list("custom_colormap", [
                                                         HOPIUM_PALETTE["blue"],
                                                         HOPIUM_PALETTE["hopium"],
                                                         HOPIUM_PALETTE["red"]
                                                         ])
CYBERPUNK_CMAP = mcolors.LinearSegmentedColormap.from_list("custom_colormap", [
                                                            CYBERPUNK_PALETTE["cyan"],
                                                            CYBERPUNK_PALETTE["pink"],
                                                            CYBERPUNK_PALETTE["yellow"]
                                                            ])

# Set of directories and files to skip
EXCLUDED_ITEMS = {
    # Files
    "._postpro.py", "._toolbox.py", "__init__.py", ".gitignore", ".git", ".vscode", "Allrun", "Allclean",
    # Directories
    "__pycache__", "yPlus", "wallShearStress", "system", "constant", "datasets"
}

# If a directory or a file match this regex, skip it
EXCLUDED_REGEX = r"^(?:\d+(?:\.\d+)?(?:e[-+]\d+)?|processor\d+|0\.original)$"

# Regex for paths to the postProcessing directory
POSTPRO_DIR = "/postProcessing/"

# Regex for the log files
LOG_REGEX = r"log\.[a-zA-Z]+Foam(?:\.\d+(?:s)?)?"

# Default root directory
DEFAULT_DIR = Path.cwd()
            # Path("/" + os.path.join(*os.getcwd().split("/")[:5]))

# ASCII format for the progress bars
BAR_FORMAT = "{desc}: \033[1m[\033[0m{bar}\033[92m{percentage:3.0f}%\033[0m\033[1m]\033[0m  "

# Range for average
RNG = 500
# %%