##############################################################################
# Project CapCurveEx
# Script: main.py
# Written by Trevor Yeow
# This script was written to provide an example for using capcurveextraction.py
##############################################################################

from capcurveextraction import capcurvemaster
import numpy as np

"""
-----------------------------------------
DETAILED NOTES
-----------------------------------------

This script provides an example of how to implement capcurveextraction.py. Users can either modify this script to suit
their own needs, or create their own script to perform their calculations.

In this example, data from an E-defense test of a 3-story disaster management center performed at E-Defense is adopted.
The E-Defense test used an artificial record created to have the same spectral acceleration demands as required by the 
Japanese Building Standard Law while having the phase of the Kobe earthquake. The record was applied 5 times with 
varying scale factors. This example only considered the first application using a 150%-scaled input. At this stage, the
building had already incurred slight/minor damage from previously applied excitations. Further information on this
test is available in the following:

    "Yeow TZ, Kusunoki K, Nakamura I, Hibino Y, Fukai S and Safi WA. (2021). E-defense shake-table test of a 
    building designed for post-disaster functionality. Journal of Earthquake Engineering. 
    https://doi.org/10.1080/13632469.2020.1865219"
   
"""

##########################

# Specifying datafile and unit conversion
"""
For this example, the acceleration data is stored in a .txt file. There are four columns of data, with the left column 
representing the ground floor, and each subsequent column representing the next floor up. Each row represents each
individual data step, with a time increment of 0.01s between each. This is represented by the "dT" parameter. The 
acceleration data is in units of gal (i.e., 981 cm/s/s). The acceleration data matrix to be inputted into the capacity
curve extraction function must be in this format. The capcurveextraction.py script assumes that the input data 
is in units of g, so the accelerations need to be multiplied by 1/981. This is presented by the "AccConvert" parameter. 
To import the data into Python, we have used "loadtxt" command as part of the "numpy" module. Due to this reason, 
"numpy" needs to be imported to run this example properly. If the user has other methods of incorporating their data 
that does not require the "numpy" module, then this does not need to be imported in this example.
"""
filename = 'ALAB-Center-150-1.txt'  # Specify input filename (with directory if needed)
AccConvert = 1/981                  # Converting acceleration back into units of g
data = np.loadtxt(filename, delimiter='\t')*AccConvert     # Read data file
dT = 0.01                           # Timestep between each datapoint (in seconds)



# Mass vector
"""
The mass vector is an optional input. If the user chooses to specify this themselves, then it should be formatted as a
row vector, with each entry representing the corresponding floor level of the acceleration data. For example, in this
case, the ground floor mass is 0, while each subsequent floor level was 740, 720 and 520. It should be noted that the
actual value of mass and the units do not matter since it ends up being normalized in the calculation. However, it is
important that the proportion of masses is accurate. For example, using [0, 1.42, 1.38, 1] instead would give the same
result. Also, the first entry is not important since it will be multiplied by the relative of the ground floor response
to itself, which is 0. One important note is that the number of entries in the mass vector must be consistent with the
number of columns of the acceleration data. If this is not consistent, an error message will be given. Finally, if the
user does not specify the mass vector themselves, then it will be automatically assumed that the distribution of 
masses with each floor level is constant (i.e., [1, 1, 1, 1] in this case if mass is not provided).
"""
Mass = [0, 740, 720, 520]           # Specify floor weights (from 1F to RF)



# Defining wavelet shape
"""
Another optional input is the mother wavelet shape to adopt in the analysis. Please refer to 
pywavelets.readthedocs.io/en/latest/ref/wavelets.html#wavelet-families for a list of wavelet families. However, please
note that a key step in the methodology was calibrated using the sym10 wavelet, though tests using db10 found that it
was also reasonably accurate. As such, we do not recommend using other wavelet types. If you choose to do so, please
do it at your own risk. If the user chooses to not specify this input, then sym10 will be assumed by default.
"""
Wavelet = 'sym10'                   # Name of wavelet



# Specifying figures to plot
"""
Another optional input is to determine which figures to plot. This is done using the "plotflag" variable. "plotflag"
has three entries, with each entry corresponding to a different figure. If a given entry has a value of 1, then the
corresponding figure will be plotted. If it has any other values, the corresponding figure will not be plotted. The
figures are as follows:
    plotflag[0] = 1: plot individual rank hysteretic response
    plotflag[1] = 1: plot tentative hysteretic response
    plotflag[2] = 1: plot final extracted backbone curve
"""
plotflag = (0, 0, 0)                # Plot flags



# Specifying data to export
"""
The final optional input is to determine which data to export. There are five possible types of data output. If the 
user wants a specific data to be exported, they simply have to specify the output file name. The exportable data are as
follows:
outrankd:   Name for file containing representative response for each individual rank information
outrank:    Name for file containing properties of each individual rank
outcombd:   Name for file containing floor response considered selected ranks only
outhyst:    Name for file containing tentative representative displacement-acceleration hysteresis
outscurve:  name for file containing skeleton curve
"""
outrankd = 'OutRankD.csv'
outrank = 'OutRankP.csv'
outcombd = 'OutCombD.csv'
outhyst = 'OutTHyst.csv'
outscurve = 'OutSCurve.csv'



# Executing program to perform capacity curve extraction
"""
To run the program, the user must firstly import capcurvemaster from capcurveextraction.py. When specifying the
function, "data" and "dT" which are the floor acceleration data and the sampling timestep, respectively, must be
provided. This alone is sufficient to run the program without any further inputs. However, it will assume that
mass is evenly distributed on all floors and will automatically use sym10 as the mother wavelet, and additionally
there will be no plotting or exporting of data. See the following example.
"""
OUT1 = capcurvemaster(data, dT)

"""
If more inputs are sought, then the variable name must be specifically specified in the function command. NOTE THAT
THESE ARE CASE SENSITIVE AND WILL NOT WORK OTHERWISE. An example containing every single input is shown below. Note 
that these do not need to be in the order shown
"""
OUT2 = capcurvemaster(data, dT, Mass=Mass, Wavelet=Wavelet, plotflag=plotflag, outrankd=outrankd, outrank=outrank,
                      outcombd=outcombd, outhyst=outhyst, outscurve=outscurve)

"""
The user does not need to specify all optional inputs. Furthermore, there is no need to predefine each parameter as
shown in the above examples. Additionally, there is also no strict need for the assigned variable to be identical
to the variable name. For example, if the user wants to use the default mass vector, a db10 mother wavelet, and
only export the final capacity curve, the command can be as follows:
"""
newwavelet = 'db10'
OUT3 = capcurvemaster(data, dT, Wavelet=newwavelet, outscurve='OutSCurveV3.txt')

"""
The output from this function (e.g., OUT1, OUT2 and OUT3) is the list of decomposition levels selected to reconstruct
the acceleration signals to derive the capacity curve. However, this output is optional. The user can omit this and
simply specify the following to obtain the same output as the previous example:
"""
capcurvemaster(data, dT, Wavelet='db10', outscurve='OutSCurveV4.txt')

"""
If the user wants to display the selected decomposition levels, simply use the "print" command as follows:
"""
print(OUT1)
