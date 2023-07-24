# Import a module

To import the module in a Python console, use:

<div style="background-color: #F0F0F0;">

```python
# Import the module in the script 
# and create the 'pp' alias
import postpro as pp 
```
</div>

# Arguments

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

- **Kwargs:** placeholder name for **optional** keyword arguments passed in the function call. They are represented by **kwargs in the function definition.
  
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

The functions of the **postpro** module use these 3 types of arguments to work correctly. They should be written in the following order:

```python
# Generic function call using positional arguments, keyword-only arguments and kwargs 
func(positional args, keyword-only args, kwargs)
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

The arguments compatible with regex are *run*, *specdir*, *probe* and *usecols*.