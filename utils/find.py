import pandas as pd
import re

from pathlib import Path
from functools import partial

import utils.constants as cst
# * ===================================================================================================

def get_labels(database: pd.DataFrame, 
                        directory: str, 
                        category: str) -> list:
    """
    Get a list of post-processing labels for a given directory and category.

    Args:
        df (pandas.DataFrame): DataFrame containing post-processing directories and labels.
        directory (str): Directory to search for labels.
        category (str): Category of labels to retrieve, 'in_file' or 'postpro'.

    Returns:
        List[str]: List of labels for the given directory and category.
    """
    # Use hierarchical indexing to retrieve all labels for the given directory and category
    labels = database.loc[(directory, category), :]

    # Filter out any NaN values from the resulting Series
    filtered_labels = [c for c in labels.to_numpy() if not pd.isna(c)]

    return filtered_labels

# * ===================================================================================================

def csv_postpro_to_df() -> pd.DataFrame:
    """
    Convert a CSV file containing post-processing directories and labels into a Pandas DataFrame.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with hierarchical indexing by directory and label type.
    """
    dirname = Path(__file__).parents[1]
    csv_path = dirname / "postpro_directories.csv"

    # Convert the csv file into a DataFrame while filling the empty cells with ""
    csv_df: pd.DataFrame = pd.read_csv(csv_path, sep=';').fillna(pd.NA)

    # Fill missing directory values using the most recent non-null value
    csv_df['Directories'] = csv_df['Directories'].fillna(method='ffill')

    # Set hierarchical indexing by directory and label type
    csv_df = csv_df.set_index(['Directories', 'Label Type'])

    return csv_df

# * ===================================================================================================

def label_names(fpath: Path) -> dict:
    if fpath.parents[1].name in ["probes", "residuals"]:
        # If the file is a probes or residuals file, the labels are determined based
        # on the format of the file.
        with fpath.open() as f:
            if fpath.name == "U":
                # If the file is a velocity file, the labels are composed of the time
                # and the velocity components.
                for line in f:
                    if line.startswith('# Time'):
                        file_labels = line.strip().split()[2:]
                        postpro_labels = ['Time'] + [pnum + i for pnum in file_labels for i in (' - Ux', ' - Uy', ' - Uz')]
                        break
            else:
                # If the file is not a velocity file, the labels are composed of the
                # time and the variables in the file.
                for line in f:
                    if line.startswith('# Time'):
                        file_labels = line.strip().split()[1:]
                        postpro_labels = file_labels
                        break
    else:
        # If the file is not a probes or residuals file, the labels are obtained from
        # a CSV file that links the postpro dirs to the corresponding labels.
        postpro_dir = fpath.parents[1].name
        csv_df = csv_postpro_to_df()
        _partial_labels = partial(get_labels, database=csv_df, directory=postpro_dir)
        file_labels, postpro_labels = _partial_labels(category='in_file'), _partial_labels(category='postpro')

    # Return a dictionary containing the original file labels and the post-processing labels.
    return {
        'file_labels': file_labels,
        'postpro_labels': postpro_labels
    }
    
# * ===================================================================================================
    
def find_paths(runs: str, *,
                data_type: str, 
                root_dir: Path = cst.DEFAULT_DIR,
                **kwargs) -> list:
    
    target_pattern = re.compile(runs)
    stack = [root_dir]
    output_paths = []
    pattern_excluded_dir = re.compile(cst.EXCLUDED_REGEX)
    pattern_postpro_dir = re.compile(cst.POSTPRO_DIR)
 
    # Loop while the stack is not empty
    while stack:
        dirpath = stack.pop()

        # * =============================== DIRS ===============================
        
        if data_type == 'dir':

            # Filter items to search
            filtered_items = [
                e for e in dirpath.iterdir() # Item in dirpath
                if e.is_dir() # Item is a dir
                and e.name not in cst.EXCLUDED_ITEMS # Item's name not in excluded list
                and not bool(re.search(pattern_excluded_dir, e.name)) # Item's name not in excluded pattern 
            ]

            # Determine if the entry goes to the output paths or to the stack
            for entry in filtered_items:
                is_match = re.search(target_pattern, entry.name)
                if bool(is_match):
                    output_paths.append(entry)
                else:
                    stack.append(entry)

        # * =============================== FILES ===============================

        elif data_type == 'file':

            # Filter items to search
            filtered_items = [
                e for e in dirpath.iterdir() # Item in dirpath
                if e.name not in cst.EXCLUDED_ITEMS # Item's name not in excluded list
                and bool(re.search(pattern_postpro_dir, str(e))) # Item's name not in excluded pattern 
            ]

            # Determine if the entry goes to the output paths or the stack
            for entry in filtered_items:
                if entry.is_file():

                    # If the entry matches the pattern or is a .dat file
                    if bool(re.search(target_pattern, entry.name)):
                        output_paths.append(entry)
                    elif entry.suffix == ".dat":
                        label_dict = label_names(entry)

                        # If a specific label is searched in the file
                        if 'search' in kwargs:
                            to_search = kwargs.get('search')
                            label_list = label_dict.get('postpro_labels')
                            lab_is_match = bool({lab for lab in label_list if bool(re.search(to_search, lab))})
                            if lab_is_match:
                                output_paths.append(entry)

                        # By default, without more info, add the .dat file to the output paths
                        else:
                            output_paths.append(entry)

                # If item is a dir, add it to the stack
                else:
                    stack.append(entry)
    return sorted(output_paths)

# * ========================= PARTIAL FUNCTIONS =========================

find_dirs = partial(find_paths, data_type='dir')
find_files = partial(find_paths, data_type='file')

# * ===================================================================================================

def find_runs(runs: str, *, root_dir: Path = cst.DEFAULT_DIR, **kwargs) -> list:
    
    # Find all the dirs in the root dir
    all_dirs = find_dirs(runs, root_dir=root_dir, **kwargs)
    # Keep the dirs containing a 'system' dir and a 'constant' dir
    run_dirs = [d for d in all_dirs if Path(d / "system").is_dir() and Path(d / "constant").is_dir()]

    return run_dirs

def find_logs(run_path: Path) -> list:
    """
    Given a path to a directory, returns a list of all log files in the directory and its 'logs' subdirectory.

    Args:
        run_path (str): The path of the directory to search for log files.

    Returns:
        list: A list of all log files in the directory and its 'logs' subdirectory.
    """
    # Initialize an empty list to store the log files.
    log_files = []
    filtered_log_files = []
    pattern = re.compile(cst.LOG_REGEX)

    # Search for log files in the run_path directory.
    log_files += [f for f in run_path.iterdir() if re.search(pattern, f.name)]

    # Check if the 'logs' subdirectory exists in the run_path directory.
    log_dir = run_path / "logs"
    if log_dir.is_dir():
        # If the 'logs' subdirectory exists, search for log files in it.
        log_files += [f for f in log_dir.iterdir() if re.search(pattern, f.name)]

    # Remove the 'log.potentialFoam' file from the log_files list if it exists.
    filtered_log_files = [log for log in log_files if log.name != 'log.potentialFoam']

    # Return the log_files list.
    return filtered_log_files