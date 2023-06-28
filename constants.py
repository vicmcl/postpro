# %%
import os
import matplotlib.colors as mcolors

# Colors palettes
CYBERPUNK_PALETTE = {
    'pink': '#ff6090',
    'yellow': '#f7d038',
    'cyan': '#00ffff',
    'orange': '#ffa270', 
    'green': '#7cff8c',
    'purple': '#9b4dff',
    'indigo': '#5c5cff',
    'teal': '#00c8a9',
}

HOPIUM_PALETTE = {
    'hopium':'#00e6b8',
    'red':'#ff6e68',
    'blue':'#6798c6',
    'orange':'#ffa23a',
    'purple':'#b36dbf',
    'green':'#2eb872',
    'pink': '#ff8ab8',
    'yellow': '#ffe94a',
}

### COLORMAP HOPIUM ###

# Define the colors of the gradient
COLORS = [
    '#e0332b',
    '#eb7d78',
    '#d4d4d7',
    '#09ffc7',
    '#00c699'
]

# Create a custom colormap using the defined colors
HOPIUM_CMAP = mcolors.LinearSegmentedColormap.from_list('custom_colormap',
                                                        [HOPIUM_PALETTE['blue'],
                                                         HOPIUM_PALETTE['hopium'],
                                                         HOPIUM_PALETTE['red']])
CYBERPUNK_CMAP = mcolors.LinearSegmentedColormap.from_list('custom_colormap',
                                                           [CYBERPUNK_PALETTE['cyan'],
                                                            CYBERPUNK_PALETTE['pink'],
                                                            CYBERPUNK_PALETTE['yellow']])

# Set of directories and files to skip
EXCLUDED_ITEMS = {
    # Files
    '._postpro.py', '._toolbox.py', '__init__.py', '.gitignore', '.git', '.vscode', 'Allrun', 'Allclean',
    # Directories
    '__pycache__', 'yPlus', 'wallShearStress', 'system', 'constant',
}

# If a directory or a file match this regex, skip it
EXCLUDED_REGEX = r'^(?:\d+(?:\.\d+)?(?:e[-+]\d+)?|processor\d+|0\.original)$'

# Regex for paths to the postProcessing directory
POSTPRO_REGEX = r'\/run\d+\w*/postProcessing'

# Regex for the log files
LOG_REGEX = r'log\.[a-zA-Z]+Foam(?:\.\d+(?:s)?)?'

# Default root directory
DEFAULT_DIR = '/' + os.path.join(*os.getcwd().split('/')[:5])

# ASCII format for the progress bars
BAR_FORMAT = '{desc}: \033[1m[\033[0m{bar}\033[92m{percentage:3.0f}%\033[0m\033[1m]\033[0m  '

# Range for average
RNG = 500
# %%