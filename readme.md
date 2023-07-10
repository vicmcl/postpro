# Python syntax

In Python, there are multiple types of arguments possible when calling a function:
- **Positional arguments:** they are passed based on their position in the function call. They are defined without being explicitly named in the function call.
Example:

<div style="background-color: #F0F0F0;">

```python
# Function using the positional argument 'name'
def print_info(name):
    print('Name passed: ' + name)

# Function call
print_info('John')
>>> Name passed: John
```
</div>

- **Keyword-only arguments:** they can only be specified by their parameter name during a function call. They cannot be passed as a positional argument and must be explicitly assigned using their parameter name. They are called after the * in the function definition.
Example:

<div style="background-color: #F0F0F0;">

```python
# Function using the positional argument 'name' 
# and the keyword-only argument 'surname'
def print_info(name, *, surname):
    print('Complete name: ' + name + ' ' + surname)
    
# Function call
print_info('John', surname='Doe')
>>> Complete name: John Doe
```
</div>

- **Kwargs:** placeholder name for a variable-length dictionary of optional keyword arguments passed in the function call. They are represented by **kwargs in the function definition.
Example:

<div style="background-color: #F0F0F0;">

```python
# Function using the positional argument 'name', 
# the keyword-only argument 'surname' 
# and kwargs to print additional information
def print_info(name, *, surname, **kwargs):
    print('Complete name: ' + name + ' ' + surname)
    for k in kwargs.keys():
        print(k + ': ' + kwargs.get(k))

# Function call
print_info('John', surname='Doe', Age='85',
           Nationality='chinese', Birthday='25/12')
>>> Complete name: John Doe
    Age: 85
    Nationality: chinese
    Birthday: 25/12
```
</div>

The functions of the **postpro** module use these 3 types of arguments to work correctly. To import the module in a Python console, use:

<div style="background-color: #F0F0F0;">

```python
# Import the module in the script 
# and create the 'pp' alias
import postpro as pp 
```
</div>

# Regex

Some arguments of the functions in the **postpro** module can be written as a *regular expression* (regex) to match a particular pattern when looking for a directory or a file name. 

Useful regex syntax:

| Syntax | Description                                    | Example pattern | Examples match                   | Examples non-match | 
| :-     | :-                                             | :-              | :-                               | :-                 |
| $      | Match end of line                              | '01$'           | run0**01** run2**01**            | run011             |
| ^      | Match start of line                            | '^force'        | **force**s **force**Coeffs       | RAD_forces         |
| .*     | Match any character(s)                         | 'l.*'           | C**l** C**l(f)**                 | Cd                 |
| [xy]   | Match at least one of the bracketed characters | '[234]'         | run00**2** run0**34**            | run005             |
| x\|y   | Match at least one of the groups of characters | '56\|78'        | run0**56** run**78**0            | run067             |

The arguments compatible with regex are *target*, *specdir*, *probe* and *usecols*. For more information about regex, see the cheat sheet.

# Functions list

## plot_data
This function creates a XY plot where the X axis represents the timesteps (time (s) or iterations) and the Y axis represents the values of a given variable evolving during the simulation.

<details><summary>More about <b>plot_data</b></summary>
<br>

**Positional arguments:** 
- run (*str*): the run(s) to get the data from.

**Keyword-only arguments:**
- specdir (*str*): the postProcessing directory to get the data from.
  
**Kwargs**: see kwargs section

Example:

<div style="background-color: #F0F0F0;">

```python
# Plot the data found in run001/postProcessing/exhaustVolFlowRate
plot_data('001', specdir='exhaustVolFlowRate')
```
</div>

---
</details>

## plot_residuals

This function creates an XY plot where the X axis represnets the timesteps (time (s) or iterations) and the Y axis represents the residuals evolution on a log scale.

<details><summary>More about <b>plot_residuals</b></summary>
<br>

**Positional arguments:** 
- run (*str*): the run(s) to get the data from.
  
**Kwargs**: see kwargs section

Example:

<div style="background-color: #F0F0F0;">

```python
# Plot the residuals of the run001 simulation
plot_residuals('001')
```
</div>

---
</details>

## plot_probes

This function creates an XY plot where the X axis represnets the timesteps (time (s) or iterations) and the Y axis represents the values of a given probe.

<details><summary>More about <b>plot_probes</b></summary>
<br>

**Positional arguments:** 
- run (*str*): the run(s) to get the data from.

**Keyword-only arguments:**
- probe (*str*): name the probe(s) to get the data from (*'p'*, *'U'*, *'k'* etc).

**Kwargs**: see kwargs section

Example:

<div style="background-color: #F0F0F0;">

```python
# Plot the data found in run001/postProcessing/probes/alpha 
plot_probes('001', probe='alpha')
```
</div>

---
</details>

## bar_chart

This function creates a bar plot to compare one or more variables between different runs or different postProcessing directories.

<details><summary>More about <b>bar_chart</b></summary>
<br>

---
</details>

## recap_sim_df

This function returns a DataFrame (table with headers and indexed rows) gathering the basic information about a simulation.

<details><summary>More about <b>recap_sim_df</b></summary>
<br>

**Positional arguments:** 
- run (*str*): the run to get the data from.

**Keyword-only arguments:**
- geometry_name (*str*): the name of the geometry that is used in the simulation.
  
Example:

<div style="background-color: #F0F0F0;">

```python
# Plot the data found in run001/postProcessing/probes/alpha 
recap_sim_df('001', geometry='some_geometry.stl')
```
</div>

The fields automativally added in the DataFrame are:

- date of the start of the simulation
- project
- run
- user
- workstation
- geometry
- clock time
- turbulence model
- iterations or simulated time
- number of processors
---

</details>

## kwargs
<details><summary>More about <b>kwargs</b></summary>
<br>

The list of optional keyword arguments is:
- usecols (*list* or *string*): indicate specific columns to be plotted from the data file:
  
<div style="background-color: #F0F0F0;">

```python
# Plot the 'Cd' data in run001/postProcessing/forceCoeffs
plot_data('001', specdir='forceCoeffs', usecols='Cd')
```
</div>

- skipstart & skipend (*integer*): indicate the number of iterations to skip at the start or the end of a simulation for a plot:
  
<div style="background-color: #F0F0F0;">

```python
# Plot the residuals in run001/postProcessing/residuals 
# while skipping the first 20 and the last 50 iterations
plot_residuals('001', skipstart=20, skipend=50)
```
</div>

- fancyplot or fp (*boolean*): indicate whether or not the LateX font is used:

<div style="background-color: #F0F0F0;">

```python
# Plot the data in run001/postProcessing/forceCoeffs 
# while using the LateX font
plot_data('001', specdir='forceCoeffs', fp=True)
```
</div>

- search (*str*): indicate a specific variable to look for when filtering the data files. This keyword does not specify the columns to be plotted, i.e. it is only used to filter files based on the variables they contain:

<div style="background-color: #F0F0F0;">

```python
# Plot the data in every 'inlet' postProcessing directories
# containing a file with at least one velocity variable 'U'
plot_data('001', specdir='inlet', search='U')
```
</div>
</html>
