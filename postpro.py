import constants as cst
import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import re
import shutil

from datetime import timedelta
from file_read_backwards import FileReadBackwards
from functools import partial
from getpass import getuser
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from prettytable import PrettyTable
from re import compile, search, match
from scipy.fft import fft, fftfreq
from socket import gethostname
from tqdm import tqdm
from warnings import warn

# Args for the figures
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['lines.markeredgewidth'] = 1
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.linewidth'] = 1

# Backend
matplotlib.use('QtAgg')

# Use seaborn to generate plots
sns.set()
sns.set_style("whitegrid")

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

# %% ===================================================================================================

def _find_logs(run_path: str) -> list:
    """
    Given a path to a directory, returns a list of all log files in the directory and its 'logs' subdirectory.

    Args:
        run_path (str): The path of the directory to search for log files.

    Returns:
        list: A list of all log files in the directory and its 'logs' subdirectory.
    """
    # Initialize an empty list to store the log files.
    log_files: list = []
    filtered_log_files: list = []
    pattern: re.Pattern = re.compile(cst.LOG_REGEX)

    # Search for log files in the run_path directory.
    log_files += [f.path for f in os.scandir(run_path) 
                  if re.search(pattern, f.name)]

    # Check if the 'logs' subdirectory exists in the run_path directory.
    if os.path.isdir(os.path.join(run_path, 'logs')):
        # If the 'logs' subdirectory exists, search for log files in it.
        log_files += [f.path for f in os.scandir(os.path.join(run_path, 'logs')) 
                      if re.search(pattern, f.name)]

    # Remove the 'log.potentialFoam' file from the log_files list if it exists.
    filtered_log_files = [log for log in log_files if os.path.basename(log) != 'log.potentialFoam']

    # Return the log_files list.
    return filtered_log_files

# * ===================================================================================================

def _issteady(run: str) -> bool:
    log_file = _find_logs(_find_runs(run)[0])[0]
    with FileReadBackwards(log_file) as frb:
        for line in frb:
            if line.startswith("Time ="):
                if line.split()[-1].isdigit():
                    return True
                else:
                    return False
                
# * ===================================================================================================

def _csv_postpro_to_df() -> pd.DataFrame:
    """
    Convert a CSV file containing post-processing directories and labels into a Pandas DataFrame.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with hierarchical indexing by directory and label type.
    """
    dirpath = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(dirpath, "postpro_directories.csv")

    # Convert the csv file into a DataFrame while filling the empty cells with ""
    csv_df: pd.DataFrame = pd.read_csv(csv_path, sep=';').fillna(pd.NA)

    # Fill missing directory values using the most recent non-null value
    csv_df['Directories'] = csv_df['Directories'].fillna(method='ffill')

    # Set hierarchical indexing by directory and label type
    csv_df = csv_df.set_index(['Directories', 'Label Type'])

    return csv_df

# * ===================================================================================================

def _ncol(handles: list) -> int:

    max_text_length = 60
    nhandles = len(handles)
    total_length = sum(len(text) for text in handles) + 3 * nhandles
    
    if total_length > max_text_length:
        if nhandles > 6:
            ncol = 4
        else:
            ncol = max(int(nhandles / 2), int((nhandles + 1) / 2))
        
        row_index = range(0, nhandles - ncol, ncol)
        for i in row_index:
            words_in_row = [handles[k] for k in range(i, i + ncol)]
            if sum(len(word) for word in words_in_row) > max_text_length:
                ncol -= 1
                break
    else:
        ncol = nhandles
    return ncol
    
# * ===================================================================================================

def _get_postpro_labels(database: pd.DataFrame, 
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
    labels: pd.Series = database.loc[(directory, category), :]

    # Filter out any NaN values from the resulting Series
    filtered_labels: list = [c for c in labels.to_numpy() if not pd.isna(c)]

    return filtered_labels

# * ===================================================================================================

def _label_names(fpath: str) -> dict:
    if 'probes' in fpath or 'residuals' in fpath:
        # If the file is a probes or residuals file, the labels are determined based
        # on the format of the file.
        with open(fpath, 'r') as f:
            if fpath.endswith('U'):
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
        # a CSV file that maps the directories and label types to the corresponding labels.
        postpro_dir: list = fpath.split('/')[-3]
        csv_df: pd.DataFrame = _csv_postpro_to_df()
        
        file_labels: list = _get_postpro_labels(
            database = csv_df,
            directory = postpro_dir,
            category = 'in_file',
        )
        postpro_labels: list = _get_postpro_labels(
            database = csv_df,
            directory = postpro_dir,
            category = 'postpro',
        )

    # Return a dictionary containing the original file labels and the post-processing labels.
    return {
        'file_labels': file_labels,
        'postpro_labels': postpro_labels,
    }

# * ===================================================================================================
    
def _find_paths(target: str, *,
                dtype: str, 
                root_dir: str = cst.DEFAULT_DIR,
                **kwargs) -> list:
    
    target_pattern = re.compile(target)
    stack = [os.path.abspath(root_dir)]
    output_paths = []
    pattern_excluded_dir = re.compile(cst.EXCLUDED_REGEX)
    pattern_postpro_dir = re.compile(cst.POSTPRO_REGEX)
 
    # Loop while the stack is not empty
    while stack:
        dirpath: str = stack.pop()

        # * =============================== DIRS ===============================
        
        if dtype == 'dir':

            # Filter items to search
            filtered_items = [
                e for e in os.scandir(dirpath) # Item in dirpath
                if e.is_dir() # Item is a dir
                and e.name not in cst.EXCLUDED_ITEMS # Item's name not in excluded list
                and not bool(re.search(pattern_excluded_dir, e.name)) # Item's name not in excluded pattern 
            ]

            # Determine if the entry goes to the output paths or to the stack
            for entry in filtered_items:
                is_match = re.search(target_pattern, entry.name)
                if bool(is_match):
                    output_paths.append(entry.path)
                else:
                    stack.append(entry.path)

        # * =============================== FILES ===============================

        elif dtype == 'file':

            # Filter items to search
            filtered_items = [
                e for e in os.scandir(dirpath) # Item in dirpath
                if e.name not in cst.EXCLUDED_ITEMS # Item's name not in excluded list
                and bool(re.search(pattern_postpro_dir, e.path)) # Item's name not in excluded pattern 
            ]

            # Determine if the entry goes to the output paths or the stack
            for entry in filtered_items:
                if entry.is_file():

                    # If the entry matches the pattern or is a .dat file
                    if bool(re.search(target_pattern, entry.name)):
                        output_paths.append(entry.path)
                    elif entry.name.endswith('.dat'):
                        label_dict = _label_names(entry.path)

                        # If a specific label is searched in the file
                        if 'search' in kwargs:
                            to_search = kwargs.get('search')
                            label_list = label_dict.get('postpro_labels')
                            lab_is_match = bool({lab for lab in label_list if bool(re.search(to_search, lab))})
                            if lab_is_match:
                                output_paths.append(entry.path)

                        # By default, without more info, add the .dat file to the output paths
                        else:
                            output_paths.append(entry.path)

                # If item is a dir, add it to the stack
                else:
                    stack.append(entry.path)
    
    return sorted(output_paths)

# * ========================= PARTIAL FUNCTIONS =========================

_find_dirs = partial(_find_paths, dtype='dir')
_find_files = partial(_find_paths, dtype='file')

# * ===================================================================================================

def _find_runs(target: str, *,
               root_dir: str = cst.DEFAULT_DIR,
               **kwargs) -> list:
    
    # Find all the dirs in the root dir
    all_dirs =  _find_dirs(target, root_dir=root_dir, **kwargs)
    # Keep the dirs containing a 'system' dir and a 'constant' dir
    run_dirs = [d for d in all_dirs if os.path.isdir(os.path.join(d, 'system')) and os.path.isdir(os.path.join(d, 'constant'))]

    return run_dirs

# * ===================================================================================================

def _concat_data_files(file_paths: list) -> dict:
        
        data_list = []

        # Sort file paths 
        if all(f.split('/')[-2].isdigit() for f in file_paths):
            timesteps = [int(f.split('/')[-2]) for f in file_paths]
        else:
            timesteps = [float(f.split('/')[-2]) for f in file_paths]
        sorted_file_paths = [x for _, x in sorted(zip(timesteps, file_paths), key=lambda pair: pair[0])]

        # Get columns name
        cols = _label_names(sorted_file_paths[0])['postpro_labels']

        # Loop over the file paths
        for fpath in sorted_file_paths:
            pp_dir = fpath.split('/')[-3]
            timestep = fpath.split('/')[-2]

            # Parse file
            with open(fpath, 'r') as f:
                for line in f:
                    if line.startswith('# Time'):
                        break

                # Verbose
                if 'probe' in fpath:
                    print(f'Parsing {bmag}probe {os.path.basename(fpath)}{reset} at timestep {bmag}{timestep}{reset}:')
                else:
                    print(f'Parsing {bmag}{pp_dir}{reset} at timestep {bmag}{timestep}{reset}:')
                
                # Set progress bar
                pbar = tqdm(total=0, unit=' lines')

                # Split each line to store the data in a tuple
                for line in f:
                    data = tuple(line.split())
                    data_list.append(data)
                    pbar.update(1)
                pbar.close()

                #Verbose
                print('File parsed.')
        
        # If at least 2 files are concatenated, display their start timestep
        if len(file_paths) > 1:
            fmt_sep = f"{reset}, {bmag}"
            fmt_timesteps = f"{fmt_sep}".join(sorted([str(i) for i in timesteps]))
            print(f'Concatenated files at timesteps {bmag}{fmt_timesteps}{reset}.')

        return data_list, cols

# * ===================================================================================================

def _remove_NaN_columns(concat_data: tuple) -> pd.DataFrame:

    data, headers = concat_data
        
    # Check for NaN columns to remove
    # Initialize the indexes of headers and columns to drop
    idx_headers_to_drop = []
    idx_cols_to_drop = []

    # DataFrame created without the headers
    df = pd.DataFrame(data)
    
    # Loop over the columns of the DataFrame
    for i in tqdm(range(len(df.columns)),
                    bar_format=cst.BAR_FORMAT,
                    ascii=' |',
                    colour='green',
                    desc='Checking for NaN columns'):
        
        # If a column filled with NaN is found    
        if df.iloc[:, i][df.iloc[:, i] != 'N/A'].size == 0:

            # Its index is saved to remove the corresponding header and column
            idx_cols_to_drop.append(i)
            idx_headers_to_drop.append(i)

            # If the velocity vector is NaN, 
            # the Ux, Uy and Uz headers must be removed
            if 'Uy' in headers[i+1] or 'Uz' in headers[i+1]:
                idx_headers_to_drop.append(i+1)
                if 'Uz' in headers[i+2]:
                    idx_headers_to_drop.append(i+2)

    # Verbose
    if idx_cols_to_drop:
        print('NaN column(s) removed.')
    else:
        print('No column to remove.')

    # Associate the columns with their headers
    df = df.drop(idx_cols_to_drop, axis=1)
    headers = [val for i, val in enumerate(headers) if i not in idx_headers_to_drop]
    df.columns = headers
    
    return df 

# * ===================================================================================================
    
def _remove_NaN_cells(df: pd.DataFrame) -> pd.DataFrame:

    # Check for NaN cells to remove
    tqdm.pandas(bar_format=cst.BAR_FORMAT,
                ascii=' |',
                colour='green',
                desc='Checking for remaining NaN cells')
    
    # Save the initial number of rows in the DataFrame
    previous_size = df.size

    # Drop the rows with one or more NaN or None cells
    df = df[~df.progress_apply(lambda x: x.isin(['N/A', None])).any(axis=1)]

    # Verbose
    if df.size != previous_size:
        print('NaN cell(s) removed.')
    else:
        print('No cell to remove.')

    return df

# * ===================================================================================================

def _remove_parenthesis(df: pd.DataFrame) -> pd.DataFrame:

    # Check for parenthesis to remove
    cols_with_parenthesis = [col for col in df.columns if df[col][1].startswith('(') or df[col][1].endswith(')')]

    # If one or more columns contain cells with parenthesis
    if len(cols_with_parenthesis) > 0:

        # Progress bar setup
        tqdm.pandas(bar_format=cst.BAR_FORMAT,
                    colour='green',
                    ascii=' |',
                    desc='Formatting vectors')
        
        # Remove the parenthesis in the identified columns
        df[cols_with_parenthesis] = df[cols_with_parenthesis].progress_apply(lambda x: x.str.replace(r'[()]', '', regex=True))

        # Verbose
        print('Vectors formatted.')

    return df

# * ===================================================================================================

def _filter_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:

    # If one or more columns are specified by the user
    if 'usecols' in kwargs:
        usecols = kwargs.get('usecols')
        if isinstance(usecols, str):
            usecols = [usecols]

        # Filter the columns to keep
        cols_found = [col for col in df.columns for u in usecols if re.search(re.compile(u), col)]
        cols = ['Time'] + cols_found

        # If only the 'Time' column remains -> the columns specified do not exist
        if len(cols) == 1:
            raise ValueError(f"{bred}'{','.join(usecols)}'{reset}: column(s) not found.")
        
        # If at least one column of data is found
        else:
            # If not all the specified columns are found
            if len(cols) < len(usecols) + 1:
                cols_not_found = [col for col in df.columns if col not in cols]
                warn(f"{bred}'{','.join(cols_not_found)}'{reset}: column(s) not found.", UserWarning)

            # Filter the specified columns that are found
            df = df.loc[:, cols]
            print(f'Columns {bmag}{f"{reset}, {bmag}".join(df.columns[1:])}{reset} selected.')

    # If there are iterations to skip at the beginning or the end of the simulation
    if 'skipstart' in kwargs:
        skipstart = kwargs.get('skipstart')
        df = df.iloc[skipstart:,:]
        print(f"First {skipstart} iterations skipped.") # Verbose
    if 'skipend' in kwargs:
        skipend = kwargs.get('skipend')
        df = df.iloc[:-skipend, :]
        print(f"Last {skipend} iterations skipped.") # Verbose

    return df
        
# * ===================================================================================================
        
def _convert_numerical_data(df: pd.DataFrame) -> pd.DataFrame:

    # Progress bar setup
    tqdm.pandas(bar_format=cst.BAR_FORMAT,
                colour='green',
                ascii=' |',
                desc='Converting data to float')

    # If the 'Time' column contains strings representing integers
    if df.iloc[2, 0].isdigit():

        # The 'Time' column is converted to integer
        df['Time'] = df['Time'].apply(int)

        # Convert all the data to floats, except the 'Time' column
        df.iloc[:, 1:] = df.iloc[:, 1:].progress_apply(lambda x: x.astype(float))

    # If the 'Time' column contains strings representing floats, convert all the data to floats
    else:
        df = df.progress_apply(lambda x: x.astype(float))

    # Verbose
    print('Data converted.')

    return df

# * ===================================================================================================

def _files_to_df(file_paths: list, **kwargs) -> pd.DataFrame:

    out = _convert_numerical_data(
              _filter_data(
                  _remove_parenthesis(
                      _remove_NaN_cells(
                          _remove_NaN_columns(
                              _concat_data_files(file_paths)))), **kwargs))
    
    return out

# * ===================================================================================================

def _print_header(run_dirs: list) -> None:
    
    project = f'{bmag}{os.path.basename(cst.DEFAULT_DIR)}{bcyan}'
    runs_num = [os.path.basename(k) for k in run_dirs] 
    format_runs = f'{bmag}{f"{reset}, {bmag}".join(sorted(runs_num))}{bcyan}'
    title_df = pd.DataFrame({f'{reset}{bold}PROJECT{bcyan}': project,
                             f'{reset}{bold}RUN(S){bcyan}': format_runs},
                            index=['Data'])
    # Create a prettytable object
    pt = PrettyTable()

    for col in title_df.columns:
        pt.add_column(col, title_df[col].values)
        pt.align[col] = 'c'
        pt.min_width[col] = int(shutil.get_terminal_size().columns / 2) - 4

    # print the table
    print(bcyan)
    print(pt, end=f'{reset}')
    print('')
    
# * ===================================================================================================

def _get_avg(df: pd.DataFrame, *,
             rng: int,
             type_avg: str = 'final',
             **kwargs) -> dict:

    # Select all the columns except 'Time' by default
    columns = list(df.columns)[1:]

    # If one or more columns are specified with 'usecols'
    if 'usecols' in kwargs:
        usecols = kwargs.get('usecols')
        columns = [col for col in list(df.columns)[1:] 
                   if re.search(re.compile(usecols), col)]
        
    # Return a dict of the mean value of each column over rng iterations
    if type_avg == 'final':
        return {c: df.loc[:, c].tail(rng).mean() for c in columns}
    
    # Get the window of series of observations of rng size for each column
    elif type_avg == 'moving':
        windows = {c: df.loc[:, c].rolling(rng) for c in columns}
        
        # Create a series of moving averages of each window for each column
        moving_avgs = {k: windows.get(k).mean().tolist() for k in windows}
        
        # Remove null entries
        final_dict = {k: moving_avgs.get(k)[rng - 1:] for k in moving_avgs}
        return final_dict

# * ===================================================================================================

def _get_data_from_run(run_path, *,
                       specdir: str = None,
                       probe: str = None,
                       **kwargs) -> list:
    
    run_id = os.path.basename(run_path)
    file_extension = '.dat'
    pp_dirs = []

    if probe != None and specdir != None:
        raise ValueError('probe and specdir are mutually exclusive.')

    # Generic data
    if specdir != None:
        pp_dirs = _find_dirs(specdir, root_dir=run_path)
        error_dir = specdir

    # Probes
    elif probe != None:
        pp_dirs = [os.path.join(run_path, 'postProcessing/probes')]
        file_extension = probe
        error_dir = "probes"
    
    # Residuals
    else:
        pp_dirs = [os.path.join(run_path, 'postProcessing/residuals')]
        error_dir = "residuals"

    # Loop over the postpro dirs
    for pp in pp_dirs:

        # ! If wrong path
        if not os.path.isdir(pp):
            warn(f'No {bred}{error_dir}{reset} directory found.', UserWarning)

        # Get the list of file in a given run and the unique basename(s)
        file_paths = sorted(_find_files(file_extension, root_dir=pp))
        basenames = {os.path.basename(f) for f in file_paths}

        # ! More than one basename found
        if len(basenames) > 1:
            raise ValueError(f"More than one data type selected: {', '.join(bn for bn in basenames)}")
        
        # Yield a DataFrame and the run and postpro dir info
        df = _files_to_df(file_paths, **kwargs)
        if not df.empty:
            if file_extension == '.dat':
                yield {'run_id': run_id, 'pp_dir': os.path.basename(pp) , 'df': df}
            else:
                yield {'run_id': run_id, 'pp_dir': os.path.basename(file_paths[0]) , 'df': df}
        else:
            continue

# * ===================================================================================================

def _get_unit(*, df,
              csv_df,
              pp_dir,
              probe,
              **kwargs):
    
    if probe != None:
        if 'unit' in kwargs: unit = kwargs.get("unit")
        elif probe == "p" or probe =="^p" or probe == "p$": unit = 'Pa'
        elif probe == "k" or probe =="^k" or probe == "k$": unit = 'J/kg'
        else: unit = None
    
    elif pp_dir == 'residuals':
        unit = 'Residuals'
        
    else:
        # List of all the units for the pp dir in the csv
        unit_list = _get_postpro_labels(csv_df, pp_dir, "unit")
        unit_length = len(unit_list)

        # List of all the headers for the pp dir in the csv 
        header_list = _get_postpro_labels(csv_df, pp_dir, "postpro")

        # If at least one unit label
        if unit_length > 1:
            # The chosen unit for the graph is the first one in the list (skip the Time column)
            unit = [unit_list[i] for i in range(unit_length) if header_list[i] in df.columns][1]
        else:
            unit = None

    return unit

# * ===================================================================================================

def _format_excel(file_path):
    # Load the Excel file
    workbook = load_workbook(file_path)
    sheet = workbook.active
    
    # Set font styles
    header_font = Font(name="Calibri", bold=True, size=14)
    content_font = Font(name="Calibri", size=12)
    
    # Set alignment
    alignment = Alignment(horizontal="center", vertical="center")
    
    # Set fill color
    fill_main = PatternFill(start_color="C7D1E0", end_color="C7D1E0", fill_type="solid")
    fill_data = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    
    # Set border
    border_color = "FF0000"
    thin_border = Border(top=Side(style=None), 
                         right=Side(style=None), 
                         bottom=Side(style=None), 
                         left=Side(style=None))
    
    # Format header row
    for cell in sheet[1]:
        cell.font = header_font
        cell.alignment = alignment
        cell.fill = fill_main
        cell.border = thin_border
        cell.value = cell.value.upper()
    
    # Format content rows
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = content_font
            cell.alignment = alignment
            cell.fill = fill_data
            cell.border = thin_border
            
            if isinstance(cell.value, (int, float)):
                if cell.coordinate >= 'K':
                    cell.number_format = '0.00E+00'  # Scientific notation format code
    
    # Increase header row height
    sheet.row_dimensions[1].height = 40
    
    for row in sheet.iter_rows(min_row=2):
        sheet.row_dimensions[row[0].row].height = 20
    
    # Calculate the maximum text length in each column
    max_text_lengths = {}
    print(sheet.column_dimensions['G'].width)
    for row in sheet.iter_rows(min_row=1, values_only=True):
        for column_index, cell_value in enumerate(row, start=1):
            column_letter = get_column_letter(column_index)
            text_length = len(str(cell_value))
            if column_letter not in max_text_lengths or text_length > max_text_lengths[column_letter]:
                max_text_lengths[column_letter] = text_length

    # Set the column width as 1.2 times the maximum text length
    for column_letter, max_length in max_text_lengths.items():
        column_width = (max_length * 1.2) + 2  # Add some extra padding
        sheet.column_dimensions[column_letter].width = column_width
    
    # Save the modified Excel file
    workbook.save(file_path)

# %% ===================================================================================================

def reload():
    module = importlib.import_module(__name__)
    importlib.reload(module)
    importlib.reload(cst)
    print('Reloaded.')

# * ===================================================================================================

def gather_runs(runs):
    # Convert CSV data to DataFrame
    csv_df = _csv_postpro_to_df()

    # Find runs in the specified directory
    runs_dir = []
    runs_dir += _find_runs(runs)

    # Count the number of unique run directories
    runs_nb = len({os.path.basename(r) for r in runs_dir})

    # Raise an error if no runs are found
    if len(runs_dir) == 0:
        raise ValueError(f"No run found with {bred}'{runs}'{reset}.")

    return runs_dir, runs_nb, csv_df

# * ===================================================================================================

def gather_data(runs_dir, specdir, probe, **kwargs):
    # List to store processed run DataFrames
    run_pp_df_list = []

    # Iterate over each run directory
    for run_path in runs_dir:
        print(f"\n{bmag}--------\n# {os.path.basename(run_path)}\n--------{reset}\n")
        
        # Get data from the run and add it to the list
        run_pp_df_list += [data for data in _get_data_from_run(run_path, specdir=specdir,
                                                                  probe=probe, **kwargs)]
        
        # Raise an error if no data is found for the input
        if not run_pp_df_list:
            raise ValueError("No data found with such input.")

    return run_pp_df_list

# * ===================================================================================================

def plot_time_data(ax, df, handle_prefix, frmt_legend):
    # Iterate over each column (excluding 'Time') in the DataFrame
    for col in [c for c in df.columns if c != 'Time']:
        handle = f"{handle_prefix}{col}{frmt_legend}"
        
        # Plot the data as a line plot
        sns.lineplot(data=df, x='Time', y=col, label=handle, ax=ax, linewidth=cst.LINEWIDTH)
        sns.despine(left=True)
        
        # Yield the handle for each iteration
        yield handle

# * ===================================================================================================

def plot_freq_data(ax, df, handle_prefix, frmt_legend, sampling_rate, **kwargs):
    # Iterate over each column (excluding 'Time') in the DataFrame
    for col in [c for c in df.columns if c != 'Time']:
        frmt_col = f"{handle_prefix}{col}"
        handle = f"{frmt_col}{frmt_legend}"

        print(f"Calculating FFT for {bmag}{col}{reset}...")
        
        # Calculate the FFT and frequency values
        signal_fft = fft(df[col].values)
        freqs = fftfreq(len(df[col])) * sampling_rate

        normalized_spectrum = np.abs(signal_fft) / np.max(np.abs(signal_fft))
        pos_freqs = freqs[freqs >= 0]

        if "lowpass" in kwargs:
            pos_freqs = pos_freqs[pos_freqs <= int(kwargs.get("lowpass"))]

        # Plot the frequency data as a line plot
        sns.lineplot(x=pos_freqs, y=normalized_spectrum[:len(pos_freqs)], label=handle, ax=ax, linewidth=cst.LINEWIDTH)
        
        # Yield the handle for each iteration
        yield handle

# * ===================================================================================================

def set_figure_params(probe, specdir):
    # Create a new figure and axis with specified parameters
    _, ax = plt.subplots(figsize=(12, 27 / 4))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.grid(axis='y', linewidth=0.1, color='00000')
    plt.grid(axis='x', linewidth=0)

    # Set y-axis scale to logarithmic if probe and specdir are None
    if probe is None and specdir is None:
        plt.yscale('log')

    return ax

# * ===================================================================================================

def format_legend(ax, handles):
    # Set legend properties
    ax.legend(
        loc='upper center',
        bbox_to_anchor=[0.5, -0.2],
        framealpha=1,
        frameon=False,
        ncol=_ncol(handles),
        borderaxespad=0,
        fontsize=12
    )

# * ===================================================================================================

def set_axis_labels(ax, freq=False, unit=None):
    # Set axis labels based on frequency or time data
    if freq:
        ax.set_ylabel("Normalized Amplitude", labelpad=10, fontsize=15)
        ax.set_xlabel("Frequency (Hz)", labelpad=18, fontsize=15)
    else:
        ax.set_xlabel("Iterations | Time (s)", labelpad=18, fontsize=15)
        ax.set_ylabel(unit, labelpad=10, fontsize=15)

# * ===================================================================================================

def plot_data(runs, *, specdir: str, probe: str = None, freq: bool = False, **kwargs):
    # Gather runs, run data, and CSV DataFrame
    runs_dir, runs_nb, csv_df = gather_runs(runs)
    run_pp_df_list = gather_data(runs_dir, specdir, probe, **kwargs)
    
    # Set up the figure parameters
    ax = set_figure_params(probe, specdir)
    handle_prefix = "Probe " if probe is not None else ""

    # Iterate over each processed run DataFrame
    for data in run_pp_df_list:
        run_id, pp_dir, df = data.values()
        frmt_legend = " | " + run_id if runs_nb > 1 else ""

        if freq:
            sampling_rate = len(df["Time"]) / df["Time"].iloc[-1]
            set_axis_labels(ax, freq=True)
            print(f"\n{bmag}------------\n# FFT {run_id}\n------------{reset}\n")
            
            # Plot frequency data and yield handles
            handles = [h for h in plot_freq_data(ax, df, handle_prefix, frmt_legend, sampling_rate, **kwargs)]
        else:
            unit = _get_unit(df=df, pp_dir=pp_dir, probe=probe, csv_df=csv_df, **kwargs)
            set_axis_labels(ax, unit=unit)
            
            # Plot time data and yield handles
            handles = [h for h in plot_time_data(ax, df, handle_prefix, frmt_legend)]

    if unit is None:
        plt.gca().set_ylabel(None)
    
    # Format the legend and set the title if specified
    format_legend(ax, handles)
    if 'title' in kwargs:
        title = kwargs.get('title')
        ax.set_title(title, fontsize=20)

    print('\nDisplaying the figure...\n')
    
    # Adjust figure layout and display the figure
    plt.tight_layout()
    plt.show()

# * ========================= PARTIAL FUNCTIONS =========================

plot_probes = partial(plot_data, specdir=None)
plot_residuals = partial(plot_data, specdir=None, probe=None)

# * ===================================================================================================

# TODO BAR CHART ===================================================================================================

# def bar_chart(target, *,
#               rng: int = cst.RNG,
#               specdir: str = None,
#               probe: str = None,
#               **kwargs) -> None:
    
#     # ! Mutual exclusion of arguments
#     if probe != None and specdir != None:
#         raise ValueError('probe and specdir are mutually exclusive.')

    
#     # Find the run(s) path
#     run_dirs = []
#     if isinstance(target, str):
#         target = [target]
#     for tar in target:
#         run_dirs += _find_runs(tar)

#     # ! If no run found
#     if len(run_dirs) == 0:
#         raise ValueError(f"No run directory found with this list.")
    
#     # Else, print the table showing the project and runs
#     else:
#         _print_header(run_dirs)
        
#         # If no probe, legend not modified
#         if probe == None:
#             lgd = ''
#             sig_list = _get_sig_list(run_dirs,
#                                         specdir=specdir,
#                                         **kwargs)
            
#         # If probe, "Probe #" added to the legend 
#         else:
#             lgd = f'Probe{space}'
#             sig_list = _get_sig_list(run_dirs,
#                                         probe=probe,
#                                         **kwargs)  

#         # Initialization
#         handles, xlabel = [], []
#         df_mean = pd.DataFrame()
#         _, ax = plt.subplots(figsize=(12, 27/4))
#         run_number = len({sig['run_id'] for sig in sig_list})
#         pp_dir_number = len({sig['pp_dir'] for sig in sig_list})
#         xpos = np.arange(len(sig_list))
#         ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 0))
        
#         # Loop over the datasets to be plotted
#         for sig in sig_list:
#             run_id = sig['run_id'] 
#             pp_dir = sig['pp_dir']
#             df = sig['df'].iloc[:, 1:] # remove time column
            
#             # Initialize a dict representing the mean data for a run/pp_dir combination
#             mean_dict = {'run': run_id, 'pp_dir': pp_dir}
#             mean_dict.update({col: df[col].tail(rng).mean() for col in df.columns})
            
#             # Add a new row of mean values in df_mean for each run/pp_dir combination
#             df_mean = pd.concat([df_mean, pd.Series(mean_dict).to_frame().T])
                            
#             # Format strings for LateX font
#             format_string = lambda x: x.replace('_', underscore).replace(' ', space)
#             frmt_run = format_string(run_id)
#             frmt_pp_dir = format_string(pp_dir)
            
#             # Format xlabel with pp_dir and/or run_id
#             if run_number > 1 and pp_dir_number > 1:  
#                 frmt_legend = f'{marker}{frmt_pp_dir}{sep}run{frmt_run}{marker}'
#             elif pp_dir_number > 1:
#                 frmt_legend = f'{marker}{frmt_pp_dir}{marker}'
#             else:
#                 frmt_legend = f'{marker}run{frmt_run}{marker}'
#             xlabel.append(frmt_legend)
            
#         # Set a multi index with the run/pp_dir combination
#         df_mean = df_mean.set_index(['run', 'pp_dir'])

#         # Set the width of the rectangles
#         width = 1.5 / (len(xpos) * len(df.columns))

#         # Loop over the columns to plot each series of data with their handle
#         for i, col in enumerate(df_mean.columns):
#             handle = f'{marker}{lgd + col}{marker}'
#             handles.append(handle)
#             rect = ax.bar(xpos + i * width,
#                             df_mean[col],
#                             width = width,
#                             label = handle)
#             if 'fancyplot' in kwargs or 'fp' in kwargs:
#                 frmt = '${:.2e}$'
#             else:
#                 frmt = '{:.2e}'
#             ax.bar_label(rect, padding=3, fmt=frmt)
            
#         # Plot parameters
#         ax.legend(loc='upper center',
#                     bbox_to_anchor = [0.5, -0.1],
#                     framealpha = 1,
#                     frameon = False,
#                     ncol = _ncol(handles),
#                     borderaxespad=0,
#                     fontsize=12)
        
#         # If a unit is specified for the y axis
#         if 'unit' in kwargs:
#             ax.set_ylabel(f'{marker}{kwargs.get("unit")}{marker}', labelpad=10)

#         # If a title is specified
#         if 'title' in kwargs:
#             title = format_string(kwargs.get('title'))
#             ax.set_title(f'{marker}{title}{marker}', fontsize=20, fontweight='bold')
            
#         # Set the xticks at the center of the grouped rectangles
#         ax.set_xticks(xpos + width * (len(df_mean.columns) - 1) / 2, xlabel, fontsize=15)
        
#         # Verbose
#         print('\nDisplaying the figure...\n')
        
#         plt.tight_layout()
#         plt.show()

# TODO BAR CHART ===================================================================================================

# * ===================================================================================================

# TODO PLOT TIME ===================================================================================================

# def plot_time(target, *, x='iterations', skipstart=10, **kwargs):
#     col1 = cst.HOPIUM_PALETTE['hopium']
#     col2 = cst.HOPIUM_PALETTE['red']

#     if x == 'iterations' or x == 'time':
#         fig, ax = plt.subplots(figsize=(12, 27/4))
#         ax_cumul = ax.twinx()
#     elif x =='both':
#         fig, (ax_time, ax_iter) = plt.subplots(1, 2, figsize=(12, 27/4), sharey=False)
#         ax_time_cumul = ax_time.twinx()
#         ax_iter_cumul = ax_iter.twinx()
    
#     # Label formatting
#     marker, space, underscore, sep = _display_settings(**kwargs)
#     # Find the run(s) path
#     run_dirs = []
#     log_files = []
#     if isinstance(target, str):
#         target = [target]
#     for tar in target:
#         run_dirs += _find_runs(tar)

#     # ! If no run found
#     if len(run_dirs) == 0:
#         raise ValueError(f"No run directory found with this list.")
    
#     # If at least one run found
#     else:
#         time_pattern = compile(r'Time = ([\d.]+)s')
#         exec_pattern = compile(r'ExecutionTime\s*=\s*([\d.]+)\s*s')
#         current_iter = 0
#         prev_iter = 0
#         restarts = []
#         cumul_time = []
#         _print_header(run_dirs)
#         for run in run_dirs:
#             run_id = search('(?<=run)\d{3}\w*', run).group(0)
#             print(f'\nRun {_bmag}{run_id}{_reset}')
#             log_files += _find_logs(run)
#             data_iter = {'timestep': [], 'exec_time': []}

#             for log in log_files:
#                 print('Log: ' + log)
#                 if len(data_iter['exec_time']) > 0:
#                     current_iter = data_iter['exec_time'][-1]

#                 with open(log, 'r') as f:
#                     time_bool = False
#                     for line in f:
#                         # Patterns to find the lines starting with 'Time' and 'ExecutionTime' and extract the values
#                         time_match = match(time_pattern, line)
#                         exec_match = match(exec_pattern, line)
#                         # If a line starts with 'Time' and time_bool is False
#                         if time_match and not time_bool:
#                             # Extract the float value of Time
#                             time_value = float(time_match.group(1))
#                             # Set time_bool to True because a line starting with 'Time' has been found
#                             time_bool = True
#                         # If a line starts with 'ExecutionTime' and time_bool is True
#                         elif exec_match and time_bool:
#                             # The value of the previous execution time is kept to calculate the time difference
#                             prev_iter = current_iter
#                             # The current execution is extracted
#                             current_iter = float(line.split()[2])
#                             # Set time_bool to False to reset the boolean indicating if a 'Time' line has been found
#                             time_bool = False
#                             # The 'Time' value is added to the timestep list
#                             data_iter['timestep'].append(time_value)
#                             # If the iteration is not the first one
#                             if len(data_iter['exec_time']) > 0:
#                                 # The timestep duration is calculated and added to the exec_time list
#                                 data_iter['exec_time'].append(current_iter - prev_iter)
#                                 cumul_time.append(cumul_time[-1] + current_iter - prev_iter)
#                             # If it is the first iteration
#                             else:
#                                 # The duration of the first timestep is added to the exec_time list
#                                 data_iter['exec_time'].append(current_iter)
#                                 cumul_time.append(current_iter)
#                 restarts.append((data_iter['timestep'][-1], len(data_iter['timestep'])))
#             ax_time.scatter(data_iter['timestep'][skipstart:],
#                             data_iter['exec_time'][skipstart:],
#                             color=col1, marker='.', s=1.5)
            
#             ax_time_cumul.plot(data_iter['timestep'][skipstart:],
#                             cumul_time[skipstart:], color=col2)
            
#             ax_iter.scatter(np.arange(skipstart, len(data_iter['timestep'])),
#                             data_iter['exec_time'][skipstart:],
#                             color=col1, marker='.', s=1.5)
            
#             ax_iter_cumul.plot(np.arange(skipstart, len(data_iter['timestep'])),
#                             cumul_time[skipstart:], color=col2)

#     for r in [element[0] for element in restarts[:1]]:
#         ax_time.axvline(r, linestyle=':')
#     for p in [element[1] for element in restarts[:1]]:
#         ax_iter.axvline(p, linestyle=':')

#     ax_time.set_xlabel(f'{marker}Time{space}(s){marker}', labelpad=10)
#     ax_iter.set_xlabel(f'{marker}Iterations{marker}', labelpad=10)

#     ax_time.tick_params(labelleft=True, labelright=False,  left=True, right=False)
#     ax_time_cumul.tick_params(labelleft=False, labelright=False,  left=False, right=True)
#     ax_iter.tick_params(labelleft=False, labelright=False,  left=True, right=False)
#     ax_iter_cumul.tick_params(labelleft=False, labelright=True,  left=False, right=True)

#     for ticklabel in ax_time.get_yticklabels():
#         ticklabel.set_color(col1)
#     for ticklabel in ax_iter_cumul.get_yticklabels():
#         ticklabel.set_color(col2)

#     fig.text(-0.03, 0.35, f'{marker}Time{space}per{space}Iteration{space}(s){marker}',
#              fontsize=15, rotation=90, color='#ef476f')
#     fig.text(1, 0.42, f'{marker}Total{space}Time{space}(s){marker}',
#              fontsize=15, rotation=-90, color='#0096c7')
#     fig.tight_layout()
#     plt.show()

# TODO PLOT TIME ===================================================================================================

# * ===================================================================================================

def sim_time(run):

    # Get run and log files  
    run_path = _find_runs(run)[0]
    log_files = _find_logs(run_path)
    times_list = []

    # Parsing log files to find the line containing the last timestep
    for log in log_files: 
        with FileReadBackwards(log, encoding='utf-8') as frb:
            for line in frb:
                # Steady simulations
                if line.startswith('ExecutionTime') and _issteady(run):
                    stime = int(line.split()[-2])
                    break
                # Unsteady simulations
                elif line.startswith('Time') and not _issteady(run):
                    stime = float(line.split()[-1][:-1])
                    break
        times_list.append(stime)
    
    # Total time calculation
    total_time = sum(times_list)
    return total_time

# * ===================================================================================================

def stop_sim(run):
    """
    This function stops the simulation by modifying the 'stopAt' line in the controlDict file
    of the specified run. It finds the path to the controlDict file and opens it, 
    then it replaces the first occurrence of 'endTime' with 'writeNow' and writes the modified line
    to the temporary file. Finally, the script uses the os.replace function to replace the original
    file with the temporary file.
    
    Parameters:
    - run (str): The name of the run folder containing the controlDict file
    
    Returns:
    - None
    """
    run_path = _find_runs(run)[0]
    # Path to the controlDict file in the run
    controlDict_path = _find_files("controlDict", root_dir=run_path + '/system/')[0]
    # Copy of the controlDict file where the new line will be written
    temp_file_path = controlDict_path + ".tmp"
    # Read the controlDict file to find the line starting with "stopAt"
    with open(controlDict_path, "r") as f, open(temp_file_path, "w") as temp:
            for line in f:
                if line.startswith("stopAt"):
                    # Replace "endTime" by "writeNow" in the temp file 
                    temp.write(line.replace("endTime", "writeNow"))
                else:
                    temp.write(line)
    print('Simulation stopping...')

    # Replace the original controlDict file by the temp file
    os.replace(temp_file_path, controlDict_path)

# * ===================================================================================================

def recap_sim(runs: str, *,
              geometry_name: str = None) -> None:

    run_paths = _find_runs(runs)

    for run_path in run_paths:
        run_id = os.path.basename(run_path)
        steady = _issteady(run_id)
        
        # Find the path to the run directory (assuming the run directory is unique)
        date_check = False
        n_procs_check = False
        date = ''
        n_procs = ''
        turbulence_model = ''
        
        # Get the turbulence model from the "turbulenceProperties" file
        momentumTransport_path = os.path.join(run_path, "constant", "momentumTransport")
        with open(momentumTransport_path, 'r') as f:
            for line in f:
                if 'model' in line:
                    turbulence_model = line.split()[-1].strip(";")
                    break

        # Get the number of iterations from the "controlDict" file
        log_files = sorted(_find_logs(run_path))
        
        data_dict = {
            'Project': [os.path.basename(os.path.dirname(run_path))],
            'Run': [run_id],
            'User': [getuser()],
            'Workstation': [gethostname()],
            'Geometry': [geometry_name],
            'Clock Time': [str(timedelta(seconds=int(sim_time(run_id))))],
            'Turbulence Model': [turbulence_model],
        }

        for i, log in enumerate(log_files):
            if i == 0:
                with open(log, 'r') as f:
                    for line in f:
                        if line.startswith('Date'):
                            date = ' '.join(line.split()[-3:])
                            date_check = True
                        elif line.startswith('nProcs'):
                            n_procs = int(line.split()[-1])
                            n_procs_check = True
                        if date_check and n_procs_check:
                            break
            if i == len(log_files) - 1:
                with FileReadBackwards(log) as frb:
                    if steady:
                        for line in frb:
                            if line.startswith("Time ="):
                                data_dict['Iterations'] = int(line.split()[-1])
                                break
                    else:
                        for line in frb:
                            if line.startswith("Time ="):
                                data_dict['Simulated Time (s)'] = float(line.split()[-1][:-1])
                                break
        
        data_dict.update({'Date': [date], '# Procs': [n_procs]})

        # Add the data to the dataframe
        df = pd.DataFrame.from_dict(data_dict)

        # Move column date to the first position
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(8))
        df = df[cols].round(3)

        pt = PrettyTable()

        for col in df.columns:
            pt.add_column(col, df[col].values)
            pt.align[col] = 'c'

        print(pt)

        return df
    
        # # Get data from run
        # run_pp_df_list += [data for data in _get_data_from_run(run_path,
        #                                                           specdir = specdir,
        #                                                           probe = probe,
        #                                                           **kwargs)]
        # if not run_pp_df_list:
        #     raise ValueError("No data found with such input.")
        
        # mean_dict = {}
        # for sig in sig_list:
        #     df = sig['df']
        #     # Remove the Time column
        #     df = df.iloc[:,1:]
        #     # Initialize a dict representing the mean data for a run/pp_dir combination
        #     mean_dict.update({col: df[col].tail(rng).mean() for col in df.columns})
        #     mean_dict = {(probe + col) if col[0].isdigit() else col: value for col, value in mean_dict.items()}

        #     data_dict.update(mean_dict)

        
        
        # new_rows = pd.concat([new_rows, df], axis=0)
        
        
        # new_cols = new_rows.columns

# TODO =====================================================================================================

# def update_excel(df: pd.DataFrame, xl_path: str = '/home/victorien/ofpostpro/recap_sim.xlsx') -> None:
        
#     if os.path.isfile(xl_path):
#         existing_rows = pd.read_excel(xl_path)
#         total_rows = existing_rows
#         existing_cols = existing_rows.columns
        
#     # ! Wrong path
#     else:
#         raise ValueError('The path to the Excel file does not exist.')

#     # Find columns in new_rows that are missing in existing_df
#     missing_cols_xl = list(set(new_cols) - set(existing_cols))
#     for col in missing_cols_xl:
#         total_rows[col] = ''
#     if missing_cols_xl:
#         print('\nNew column(s) added:',
#               f'{bmag}{f"{reset}, {bmag}".join(sorted([str(i) for i in missing_cols_xl]))}{reset}.')
    
#     # Find columns in new_rows that are missing in new_rows
#     missing_cols_new_rows = list(set(existing_cols) - set(new_cols))
#     for col in missing_cols_new_rows:
#        new_rows[col] = ''

#     # Append the new rows to the existing DataFrame
#     appended_df = pd.concat([total_rows, new_rows], ignore_index=True) 
#     sorted_missing_cols = sorted(missing_cols_xl)
#     sorted_cols = existing_cols.to_list() + sorted_missing_cols
#     appended_df = appended_df[sorted_cols]
#     # Save the updated DataFrame to a new Excel file
#     appended_df.to_excel(xl_path, index=False)
    
#     _format_excel(xl_path)