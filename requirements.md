# Packages to install

## How to install

To install a new package, use:

<div style="background-color: #F0F0F0;">

```bash
$ pip install <package>
```
</div>

## Packages and versions

The code runs with Python 3.8, and errors might occur for more recent versions. To check what version is used in the terminal, use:

<div style="background-color: #F0F0F0;">

```bash
$ python3 --version
```
</div>

The necessary packages with their minimum version is listed below:

| Packages            | Version  |
| :------------------ | -------: |
| file-read-backwards | 3.0.0    |
| matplotlib          | 3.7.1    | 
| numpy               | 1.24.2   |
| openpyxl            | 3.1.2    |
| pandas              | 2.0.1    |
| prettytable         | 3.7.0    |
| seaborn             | 0.12.2   |
| scipy               | 1.10.1   |
| tqdm                | 4.65.0   |

To check the version of a package, use:

<div style="background-color: #F0F0F0;">

```bash
$ pip list | grep <package>
```
</div>

To upgrade a package to a newer version, use:

<div style="background-color: #F0F0F0;">

```bash
$ pip install <package> --upgrade
```
</div>
