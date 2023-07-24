# Functions list

## plot_data
This function creates a XY plot where the X axis represents the timesteps (time (s) or iterations) and the Y axis represents the values of a given variable evolving during the simulation.

<details><summary>More about <b>plot_data</b></summary>
<br>

**Positional arguments:** 
- run (*str*): the run(s) to get the data from.

**Keyword-only arguments:**
- specdir (*str*): the postProcessing directory to get the data from.
- csv_path (*str*): local path to postpro_directories.csv.
- freq (*bool*): set to True to plot a frequency graph. Default value = False.
  
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

## recap_sim

This function returns a DataFrame (table with headers and indexed rows) gathering the basic information about a simulation.

<details><summary>More about <b>recap_sim</b></summary>
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

- search (*str*): indicate a specific variable to look for when filtering the data files. This keyword does not specify the columns to be plotted, i.e. it is only used to filter files based on the variables they contain:

<div style="background-color: #F0F0F0;">

```python
# Plot the data in every 'inlet' postProcessing directories
# containing a file with at least one velocity variable 'U'
plot_data('001', specdir='inlet', search='U')
```

</div>

- lowpass (*int*): set the max frequency to plot on a frequency graph. Default value = 100000:
  
```python
# Plot the frequency of the force signals in run001/postProcessing/forces until 50kHz 
plot_data('001', specdir='forces', freq=True, usecols='F', lowpass=50000)
```
</div>
</html>
