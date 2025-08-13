##############################################################################
# Project: CapCurveEx
# Script: capcurveextraction.py
# Author: Trevor Yeow
# Version 0.2 (09 May 2022)
# This script was written to perform the capacity curve extraction method
##############################################################################

import math
import sys
from wrcoef import wavedec, wrcoef
import matplotlib.pyplot as plt
import numpy as np
import pywt
from numpy import mean
from scipy import signal


def capcurvemaster(data, dT, Mass=None, Wavelet=None, plotflag=None, outrankd=None, outrank=None,
                   outcombd=None, outhyst=None, outscurve=None):
    """
        -----------------------------------------
        DETAILED NOTES
        -----------------------------------------
        This function calculates the capacity curve of a building based on its response during a seismic event.
        Please refer to the manual for detailed descriptions of the calculation process. For further information
        on the development of the "rank selection algorithm", please refer to:

            "Yeow TZ and Kusunoki K. (2022). "Unbiased rank selection for automatic hysteretic response
            extraction of RC frame buildings using acceleration recordings for post-earthquake safety
            evaluations", Earthquake Engineering & Structural Dynamics, 51(3), 515-536"


        Function command
        -----------------------------------------
        capcurvemaster(data, dT)
        capcurvemaster(data, dT, ...)
        OUT = capcurvemaster(data, dT, ...)
        LowRank, HighRank = capcurvemaster(data, dT, ...)


        Required input parameters
        -----------------------------------------
        data:       Matrix containing total floor acceleration data in units of "g". Matrix should have dimensions
                    of SigLength by NumF, where SigLength is the length of each floor recording signal, and each
                    NumF is the number of floors in the building (including ground and basement floors if applicable).
                    Each subsequent column should correspond to each subsequent floor level in increasing order.
                    E.g., column [0] should be the lowest floor, while column [NumF-1] should be the roof level

        dT :        Timestep between each recorded data. This is assumed to be constant and in units of seconds


        Optional input parameters
        -----------------------------------------
        Mass:       A row of floor mass values, with each entry corresponding to the corresponding floor from
                    "data". E.g., the last entry in Mass should correspond to the mass of the roof level.
                         Note1: if this row is not provided, it will be assumed that the mass is the same on all floors
                                of the building.
                         Note2: if the user specifies this row themselves, it must have dimensions of 1 by NumF. If
                                this condition is not satisfied, the function will terminate prematurely

        Wavelet:    The mother wavelet adopted for performing the discrete wavelet transform method.
                         DEFAULT - 'sym10'

        plotflag:   A 1 by 3 entry which flags which figures should be plotted
                        plotflag[0] = 1: plot individual rank hysteretic response
                        plotflag[1] = 1: plot tentative hysteretic response
                        plotflag[2] = 1: plot final extracted backbone curve
                        Note: if any other values are used for flags, the corresponding plot will not be generated

        Name of output files (if not provided, corresponding output file is not generated)
        outrankd:   Name for file containing representative response for each individual rank information
        outrank:    Name for file containing properties of each individual rank
        outcombd:   Name for file containing floor response considered selected ranks only
        outhyst:    Name for file containing tentative representative displacement-acceleration hysteresis
        outscurve:  name for file containing skeleton curve


        Output parameters
        -----------------------------------------
        selectedranks:     (OPTIONAL) Selected ranks


        Required modules
        -----------------------------------------
        * matplotlib
        * numpy
        * pywavelets
        * scipy


        Additional code(s) from github
        -----------------------------------------
        * wrcoef (developed by Ilya Zlotnik, 2017)
        https://github.com/izlotnik/wavelet-wrcoef


        """

    #######################################################################################
    #######################################################################################
    # NOTE: The code from this point forward is based on Yeow at al. (2022).
    # !!!!!!!!MODIFY AT YOUR OWN RISK!!!!!!!!!!!!

    #######################################################################################
    # STEP1: INITIALIZING ANALYSES
    #######################################################################################

    if not Wavelet:
        Wavelet = 'sym10'

    if not plotflag:
        plotflag = (0, 0, 0)

    SigLength = len(data[:, 1])                                    # Determine length of signal
    NumF = len(data[1, :])                                         # Determine number of floors
    Nrank = pywt.dwt_max_level(SigLength, Wavelet)                 # Determine max number of levels for DWT
    Atotalrank = np.zeros((SigLength, Nrank, NumF))                # Setup total acceleration 3D matrix
    Arelrank = np.zeros((SigLength, Nrank, NumF))                  # Setup relative acceleration 3D matrix
    Vtotalrank = np.zeros((SigLength, Nrank, NumF))                # Setup total velocity 3D matrix
    Dtotalrank = np.zeros((SigLength, Nrank, NumF))                # Setup total displacement 3D matrix
    Drelrank = np.zeros((SigLength, Nrank, NumF))                  # Setup relative displacement 3D matrix
    RepArank = np.zeros((SigLength, Nrank))                        # Setup representative acceleration 2D matrix
    RepVrank = np.zeros((SigLength, Nrank))                        # Setup representative velocity 2D matrix
    RepDrank = np.zeros((SigLength, Nrank))                        # Setup representative displacement 2D matrix
    EMrank = np.zeros((SigLength, Nrank))                          # Setup effective mass 2D matrix
    RankPara = np.zeros((Nrank, 5))                                # Setup rank parameter 2D matrix
    Atotal = np.zeros((SigLength, NumF))                           # Setup total acc from selected ranks 2D matrix
    Arel = np.zeros((SigLength, NumF))                             # Setup rel acc from selected ranks 2D matrix
    Drel = np.zeros((SigLength, NumF))                            # Setup rel disp from selected ranks 2D matrix

    if not Mass:
        Mass = np.ones(NumF)

    # Break analyses if inconsistent with size of Mass vector and NumF
    if len(Mass) != NumF:
        print('!!!ERROR!!!')
        print('Size of mass vector inconsistent with number of columns of imported acceleration data')
        print('Terminating process')
        sys.exit()

    #######################################################################################
    # STEP2: DETERMINE ACCELERATION AND DISPLACEMENT RESPONSE FOR EACH RANK AND FLOOR
    #######################################################################################

    # Obtain total response for each rank and each floor
    for j in range(0, NumF):
        C, L = wavedec(data[:, j], wavelet=Wavelet, level=Nrank)
        for i in range(0, Nrank):
            # Reconstructed detailed signal for each rank and floor and store in total acceleration matrix
            Atotalrank[:, i, j] = wrcoef(C, L, wavelet=Wavelet, level=i + 1)
            # Apply trapezium rule to get total velocity matrix
            Vtotalrank[:, i, j] = signal.detrend(np.cumsum(Atotalrank[:, i, j]*981*dT))
            # Apply trapezium rule to get total displacement matrix
            Dtotalrank[:, i, j] = signal.detrend(np.cumsum(Vtotalrank[:, i, j]*dT))

    # Obtain relative response for each rank and each floor
    for j in range(0, NumF):
        # Subtract 1F accelerations from other floors
        Arelrank[:, :, j] = Atotalrank[:, :, j] - Atotalrank[:, :, 0]
        # Subtract 1F displacements from other floors
        Drelrank[:, :, j] = Dtotalrank[:, :, j] - Dtotalrank[:, :, 0]

    #######################################################################################
    # STEP3: DETERMINE ACCELERATION, VELOCITY, DISPLACEMENT, EFFECTIVE MASS FOR EACH RANK
    #######################################################################################

    for i in range(0, Nrank):
        # Obtain tentative representative displacement at each rank
        RepArank[:, i] = np.sum(np.multiply(Arelrank[:, i, :], Mass), axis=1)/sum(Mass[1:NumF])+Atotalrank[:, i, 0]
        # Obtain tentative representative displacement at each rank
        RepDrank[:, i] = np.sum(np.multiply(Drelrank[:, i, :], Mass), axis=1)/sum(Mass[1:NumF])
        # Determine effective mass
        EMrank[:, i] = np.square(abs(np.sum(np.multiply(Drelrank[:, i, :], Mass), axis=1))) /\
                   np.sum(np.multiply(np.square(abs(Drelrank[:, i, :])), Mass), axis=1)/sum(Mass[1:NumF])

        for j in range(0, SigLength-1):
            RepVrank[j+1, i] = (RepDrank[j+1, i]-RepDrank[j, i])/dT

    #######################################################################################
    # STEP4: DETERMINE KEY PROPERTIES OF EACH RANK
    #######################################################################################

    # Determine time corresponding to 5% and 75% of max(IA)
    IA = np.cumsum(np.square(data[:, 0]))
    def condition(x): return x > 0.05*max(IA)
    nmin = min(min(np.where(condition(IA))))
    def condition(x): return x > 0.75*max(IA)
    nmax = min(min(np.where(condition(IA))))

    for i in range(0, Nrank):
        RankPara[i, 0] = max(abs(RepDrank[:, i]))                           # Peak displacement for each rank
        RankPara[i, 1] = max(abs(RepArank[:, i]))                           # Peak acceleration for each rank
        RankPara[i, 2] = mean(EMrank[nmin:nmax, i])
        RankPara[i, 3], b = np.polyfit(RepDrank[:, i], -RepArank[:, i], 1)  # Best-fit slope for each rank
        RankPara[i, 4] = 0.5*sum(np.square(RepVrank[:, i])/RankPara[i, 2])*dT

    #######################################################################################
    # STEP5: RANK SELECTION
    #######################################################################################

    # Determine initial rank
    def condition(x): return x == max(RankPara[:, 4]*((RankPara[:, 1] > 0.25*max(RankPara[:, 1]))*1))
    InitialRank = min(min(np.where(condition(RankPara[:, 4]))))+1

    # Determine highest rank to select
    ParamHigh = RankPara[:, 3]*RankPara[:, 1]/RankPara[InitialRank-1, 3]/RankPara[InitialRank-1, 1]
    def condition(x): return x >= 0.01
    HighRank = max(min(np.where(condition(ParamHigh))))+1

    # Determine lowest rank to select
    ParamLow = (RankPara[:, 0] > 0.05*max(RankPara[:, 0]))*(RankPara[:, 2]>0.65*max(RankPara[:, 2]))*1
    def condition(x): return x == 1
    LowRank = min(min(np.where(condition(ParamLow))))+1

    #######################################################################################
    # STEP6: RECONSTRUCT FLOOR RESPONSE SIGNAL BASED ON SELECTED RANKS
    #######################################################################################

    for j in range(0, NumF):
        Atotal[:, j] = Atotal[:, j] + np.sum(Atotalrank[:, LowRank - 1: HighRank, j], axis=1)
        Arel[:, j] = Arel[:, j] + np.sum(Arelrank[:, LowRank - 1: HighRank, j], axis=1)
        Drel[:, j] = Drel[:, j] + np.sum(Drelrank[:, LowRank - 1: HighRank, j], axis=1)

    #######################################################################################
    # STEP7: DERIVE HYSTERETIC RESPONSE
    #######################################################################################

    # Obtain actual representative displacement of selected ranks
    RepATemp = np.sum(np.multiply(np.square(abs(Drel)), Mass), axis=1)/np.square(abs(np.sum(np.multiply(Drel, Mass), axis=1)))*\
               np.sum(np.multiply(Arel, Mass), axis=1)+Atotal[:, 0]

    # Obtain actual representative displacement of selected ranks
    RepDTemp = np.sum(np.multiply(np.square(abs(Drel)), Mass), axis=1)/np.sum(np.multiply(Drel, Mass), axis=1)

    # Obtain effective mass ratio
    EM = np.square(abs(np.sum(np.multiply(Drel, Mass), axis=1)))/np.sum(np.multiply(np.square(abs(Drel)), Mass), axis=1)/sum(Mass[1:NumF])

    # Obtain tentative representative displacement of selected ranks
    RepATempTen = np.sum(np.multiply(Arel, Mass), axis=1)/sum(Mass[1:NumF])+Atotal[:, 0]

    # Obtain tentative representative displacement of selected ranks
    RepDTempTen = np.sum(np.multiply(Drel, Mass), axis=1)/sum(Mass[1:NumF])

    #######################################################################################
    # STEP8: EXTRACT BACKBONE CURVE
    #######################################################################################

    CapCurve = np.zeros((1, 2))

    counter = 0
    for i in range (0, len(RepDTemp)):
        if EM[i] > 0.5:
            if RepDTemp[i] > max(CapCurve[:, 0]) or RepDTemp[i] < min(CapCurve[:, 0]):
                CapCurve = np.r_[CapCurve, [(RepDTemp[i], -RepATemp[i])]]

    CapCurve = CapCurve[CapCurve[:, 0].argsort()]

    #######################################################################################
    # EXTRA1: Create required plots
    #######################################################################################

    # Plotting hysteretic response for reach rank
    if plotflag[0] == 1:
        # Setting up plot
        Nrankplotx = math.ceil(np.sqrt(Nrank))                  # Determine number of figures across
        Nrankploty = round(np.sqrt(Nrank))                      # Determine number of figures down
        ymax = np.ceil(max(map(max, abs(RepArank))))          # Determine yaxis limits for all subplots
        xmax = np.ceil(max(map(max, abs(RepDrank))))    # Determine xaxis limits for all subplots

        fig, axs = plt.subplots(Nrankploty, Nrankplotx)         # Set up figure
        for iplot in range(0, Nrankploty):
            for jplot in range(0, Nrankplotx):
                if (jplot+1)+iplot*Nrankplotx<=Nrank:           # Plot rep hystersis response for ranks
                    axs[iplot, jplot].plot(RepDrank[:, (jplot+1)+iplot*Nrankplotx-1],
                                           -RepArank[:, (jplot+1)+iplot*Nrankplotx-1])
                    axs[iplot, jplot].set_xlim([-xmax, xmax])
                    axs[iplot, jplot].set_ylim([-ymax, ymax])
                    axs[iplot, jplot].text(-xmax*0.8, ymax*0.6, "Rank "+str((jplot+1)+iplot*Nrankplotx))
                    axs[iplot, jplot].grid()
                else:                                           # For empty plots, state "N/A"
                    axs[iplot, jplot].set_xlim([-xmax, xmax])
                    axs[iplot, jplot].set_ylim([-ymax, ymax])
                    axs[iplot, jplot].text(-xmax * 0.8, ymax * 0.6, "N/A")
                    axs[iplot, jplot].grid()
        for ax in axs.flat:                                     # Set x and y label
            ax.set(xlabel='Rep Disp (cm)', ylabel='Rep Acc (g)')
            ax.label_outer()

    # Plotting tentative hysteretic response
    if plotflag[1] == 1:
        plt.figure(2)
        plt.plot(RepDTempTen, -RepATempTen)
        plt.xlabel("Tentative Rep Displacement (cm)", fontsize=20)
        plt.ylabel("Tentative Rep Acceleration (g)", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        CCxlim = np.ceil(max(abs(RepDTempTen)) / 5) * 5
        CCylim = np.ceil(max(abs(RepATempTen)) / 0.5) * 0.5
        plt.xlim(-CCxlim, CCxlim)
        plt.ylim(-CCylim, CCylim)
        plt.tight_layout()
        plt.grid()

    # Plotting final capacity curve
    if plotflag[2] == 1:
        plt.figure(3)
        plt.plot(CapCurve[:, 0], CapCurve[:, 1])
        plt.xlabel("Representative Displacement (cm)", fontsize=20)
        plt.ylabel("Representative Acceleration (g)", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        CCxlim = np.ceil(max(abs(CapCurve[:, 0])) / 5) * 5
        CCylim = np.ceil(max(abs(CapCurve[:, 1])) / 0.5) * 0.5
        plt.xlim(-CCxlim, CCxlim)
        plt.ylim(-CCylim, CCylim)
        plt.tight_layout()
        plt.grid()

    if min(plotflag) == 1 or max(plotflag) == 1:
        plt.show()

    #######################################################################################
    # EXTRA2: EXPORT DATA FILES
    #######################################################################################

    if not outrankd is None:
        rankdfile = open(outrankd, 'w')
        rankdheader = 'Rank1Disp(cm), '
        for i in range(0, Nrank-1):
            rankdheader = rankdheader+("Rank"+str(i+2)+'Disp(cm), ')
        for i in range(0, Nrank):
            rankdheader = rankdheader+("Rank"+str(i+1)+'Acc(g), ')
        np.savetxt(outrankd, np.concatenate((RepDrank, -RepArank), axis=1), delimiter=',', header=rankdheader,
                   comments='')
        rankdfile.close()

    if not outrank is None:
        rankpfile = open(outrank, 'w')
        rankpheader = 'Rank, PeakDisp(cm), PeakAcc(g), MeanEffMass, Slope(g/cm), CumulativeKEnergy(cm^2/s)'
        np.savetxt(outrank, np.concatenate((np.vstack(range(1, Nrank+1)), RankPara), axis=1), delimiter=',',
                   header=rankpheader, comments='')
        rankpfile.close()

    if not outcombd is None:
        combrfile = open(outcombd, 'w')
        combrheader = 'Rel1FDisp(cm), '
        for i in range(0, NumF - 1):
            combrheader = combrheader + ("RelDisp" + str(i + 2) + 'F(cm), ')
        for i in range(0, NumF):
            combrheader = combrheader + ("RelAcc" + str(i + 1) + 'F(g), ')
        for i in range(0, NumF):
            combrheader = combrheader + ("TotalAcc" + str(i + 1) + 'F(g), ')
        np.savetxt(outcombd, np.concatenate((Drel, -Arel, -Atotal), axis=1), delimiter=',', header=combrheader,
                   comments='')
        combrfile.close()

    if not outhyst is None:
        thystfile = open(outhyst, 'w')
        thystheader = 'TentRepDisp(cm), TentRepAcc(g)'
        np.savetxt(outhyst, np.concatenate((np.vstack(RepDTempTen), np.vstack(-RepATempTen)), axis=1),
                  delimiter=',',  header=thystheader, comments='')
        thystfile.close()

    if not outscurve is None:
        scurvefile = open(outscurve, 'w')
        scurveheader = 'BCRepDisp(cm), BCRepAcc(g)'
        np.savetxt(outscurve, CapCurve, delimiter=',', header=scurveheader, comments='')
        scurvefile.close()

    selectedranks = np.hstack(range(LowRank, HighRank+1))

    return selectedranks
