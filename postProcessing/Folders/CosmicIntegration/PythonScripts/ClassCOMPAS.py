#!/usr/bin/env python3
import numpy as np
import h5py as h5
import os
import totalMassEvolvedPerZ as MPZ

class COMPASData(object):    
    """Class for COMPAS output.

    Attributes:
        path (str): Path to directory containing COMPAS h5 file.
        fileName (str): Name of COMPAS h5 file.
        lazyData (boo): Lazy data
        Mlower (float): Minimum mass when drawing from Kroupa IMF.
        Mupper (float): Maximum mass when drawing from Kroupa IMF.
        binaryFraction (float): Binary fraction
        metallicityGrid: Array of metallicities found in COMPAS output
        metallicitySystems: Array of metallicities of each system in SystemParameters
        delayTimes: Array of delay times (Myr) of each system in DoubleCompactObjects
        mass1: Array of primary masses (Msun) of each system in DoubleCompactObjects
        mass2: Array of secondary masses (Msun) of each system in DoubleCompactObjects
        DCOmask: Mask for systems of interest in DoubleCompactObjects
        allTypesMask: Mask for BBHs, BNSs, and BHNSs of interest in DoubleCompactObjects
        BBHmask : Mask for BBHs of interest in DoubleCompactObjects 
        DNSmask : Mask for DNSs of interest in DoubleCompactObjects 
        BHNSmask : Mask for BHNSs of interest in DoubleCompactObjects 
        mchirp: Array of chirp masses (Msun) of each system in DoubleCompactObjects
        q: Array of mass ratios, q = m2/m1 in DoubleCompactObjects
        pessimisticMask: Mask for systems in DoubleCompactObjects that are consistent with the pessimistic CE assumtion
        massFormedAtEachZ: Stellar mass formed at each metallicity in the COMPAS output (in contrast with the actual stellar mass evolved by COMPAS)
    """

    def __init__(
        self,
        path=None,
        fileName="COMPAS_output.h5",
        lazyData=True,
        Mlower=None,
        Mupper=None,
        binaryFraction=None,
    ):
        """Inits ClassCOMPAS

        Args:
            path (str, optional): Path to directory containing COMPAS h5 file. Defaults to None.
            fileName (str, optional): Name of COMPAS h5 file. Defaults to "COMPAS_output.h5".
            lazyData (bool, optional): Lazy data. Defaults to True.
            Mlower (float, optional): Minimum mass when drawing from Kroupa IMF. Defaults to None.
            Mupper (float, optional): Maximum mass when drawing from Kroupa IMF. Defaults to None.
            binaryFraction (float, optional): Binary fraction. Defaults to None.

        Raises:
            ValueError: If COMPAS h5 file cannot be found from the specified path and fileName
        """
        self.path = path
        self.fileName = fileName
        if self.path is None:
            print("path not set in instance of ClassCOMPAS")
        elif not os.path.isfile(path + fileName):
            raise ValueError("h5 file not found in %s" % (path + fileName))
        elif os.path.isfile(path + fileName):
            pass

        self.metallicityGrid = None
        self.metallicitySystems = None
        self.delayTimes = None
        self.mass1 = None
        self.mass2 = None
        self.DCOmask = None
        self.allTypesMask = None
        self.BBHmask = None
        self.DNSmask = None
        self.BHNSmask = None

        self.lazyData = lazyData
        self.mChirp = None
        self.q = None
        self.pessimisticMask = None

        self.Mlower = Mlower
        self.Mupper = Mupper
        self.binaryFraction = binaryFraction
        self.massFormedAtEachZ = None

        print("ClassCOMPAS: Remember to self.setZgridAndMassEvolved()")
        print("                   then  self.setCOMPASDCOmask()")
        print("                   then  self.setCOMPASData()")

    def setCOMPASDCOmask(
        self, types="BBH", withinHubbleTime=True, pessimistic=True, noRLOFafterCEE=True
    ):
        """Set masks for systems of interest in DoubleCompactObjects

        Args:
            types (str, optional): Stellar type to mask for ('BBH','BNS','BHNS','all'). Defaults to "BBH".
            withinHubbleTime (bool, optional): True if masking for DCOs that merge within a Hubble time. Defaults to True.
            pessimistic (bool, optional): True if using the pessimistic CE prescription (HG donors cannot survive a CEE). Defaults to True.
            noRLOFafterCEE (bool, optional): True if not allowing immediate RLOF post-CE. Defaults to True.

        Raises:
            ValueError: If types is not one of 'BBH','BNS','BHNS', or 'all'
        """    
        data = h5.File(self.path + self.fileName)
        fDCO = data["DoubleCompactObjects"]
        fCEE = data["CommonEnvelopes"]

        # Masks for DCO type
        maskBBH = (fDCO["Stellar_Type_1"][()] == 14) &\
                  (fDCO["Stellar_Type_2"][()] == 14)
        maskDNS = (fDCO["Stellar_Type_1"][()] == 13) &\
                  (fDCO["Stellar_Type_2"][()] == 13)
        maskBHNS = (
                (fDCO["Stellar_Type_1"][()] == 14) &\
                (fDCO["Stellar_Type_2"][()] == 13)
            ) |\
            (
                (fDCO["Stellar_Type_1"][()] == 13) &\
                (fDCO["Stellar_Type_2"][()] == 14)
            )
        maskAllTypes = maskBBH | maskDNS | maskBHNS

        if types == "BBH":
            maskTypes = maskBBH
        elif types == "BNS":
            maskTypes = maskDNS
        elif types == "BHNS":
            maskTypes = maskBHNS
        elif types == "all":
            maskTypes = maskAllTypes
        else:
            raise ValueError("type=%s not one of 'BBH', 'BNS', 'BHNS', 'all'" % (types))

        # Mask for DCOs merging within Hubble time
        if withinHubbleTime:
            maskHubble = fDCO["Merges_Hubble_Time"][()] == True
        else:
            maskHubble = np.ones(len(fDCO["Merges_Hubble_Time"][()]), dtype=bool)

        # Masks related to CEEs
        CEEseeds = fCEE["SEED"][()]
        DCOseeds = fDCO["SEED"][()]

        if pessimistic or noRLOFafterCEE:
            maskDCOCEEs = np.in1d(CEEseeds, DCOseeds)  # Mask for CEEs involved in forming DCOs
            DCOCEEseeds = CEEseeds[maskDCOCEEs]  # Seeds of CEEs involved in forming DCOs

        # Mask for DCOs formed assumming pessimistic CEE
        if pessimistic:
            optimisticFlagForDCOCEEs = fCEE["Optimistic_CE"][()][maskDCOCEEs]
            optimisticCEEseeds = np.unique(DCOCEEseeds[optimisticFlagForDCOCEEs]) # Seeds of DCOs that have optimistic CEE
            maskOptimistic = np.in1d(DCOseeds, optimisticCEEseeds)
            maskPessimistic = np.logical_not(maskOptimistic)
        else:
            maskPessimistic = np.ones(len(fDCO["ID"][()]), dtype=bool)

        # Mask for DCOs formed without RLOF immediately after CEE
        if noRLOFafterCEE:
            immediateRLOFflagforDCOCEEs = fCEE["Immediate_RLOF>CE"][()][maskDCOCEEs]
            # Seeds of DCOs that have immediate RLOF post-CEE
            immediateRLOFCEEseeds = np.unique(DCOCEEseeds[immediateRLOFflagforDCOCEEs])
            maskRLOFafterCEE = np.in1d(DCOseeds, immediateRLOFCEEseeds)
            maskNoRLOFafterCEE = np.logical_not(maskRLOFafterCEE)
        else:
            maskNoRLOFafterCEE = np.ones(len(fDCO["ID"][()]), dtype=bool)

        # Combine all the masks
        self.DCOmask = maskTypes & maskHubble & maskPessimistic & maskNoRLOFafterCEE
        self.BBHmask = maskBBH & maskHubble & maskPessimistic & maskNoRLOFafterCEE
        self.DNSmask = maskDNS & maskHubble & maskPessimistic & maskNoRLOFafterCEE
        self.BHNSmask = maskBHNS & maskHubble & maskPessimistic & maskNoRLOFafterCEE
        self.allTypesMask = maskAllTypes & maskHubble & maskPessimistic & maskNoRLOFafterCEE
        self.pessimisticMask = maskPessimistic
        data.close()

    def setZgridAndMassEvolved(self):
        """Sets metallicity grid from COMPAS output and calculates the total star-forming
        mass at each metallicity
        """      
        _, self.massFormedAtEachZ = MPZ.getMassFormedAtEachZ(
            path=self.path,
            fileName=self.fileName,
            Mlower=self.Mlower,
            Mupper=self.Mupper,
            binaryFraction=self.binaryFraction,
        )

        # Get metallicity grid
        data = h5.File(self.path + self.fileName)
        metallicityGrid = data["SystemParameters"]["Metallicity@ZAMS_1"][()]
        self.metallicityGrid = np.unique(metallicityGrid)
        data.close()

    def setCOMPASData(self):
        Data = h5.File(self.path + self.fileName)
        fDCO = Data["DoubleCompactObjects"]
        # sorry not the prettiest line is a boolean slice of seeds
        # this only works because seeds in systems file and DCO file are printed
        # in same order

        # Get metallicity grid of DCOs
        self.seedsDCO = fDCO["SEED"][()][self.DCOmask]
        initialSeeds = Data["SystemParameters"]["SEED"][()]
        initialZ = Data["SystemParameters"]["Metallicity@ZAMS_1"][()]
        maskMetallicity = np.in1d(initialSeeds, self.seedsDCO)
        self.metallicitySystems = initialZ[maskMetallicity]

        self.delayTimes = np.add(
            fDCO["Time"][()][self.DCOmask], fDCO["Coalescence_Time"][()][self.DCOmask]
        )
        self.mass1 = fDCO["Mass_1"][()][self.DCOmask]
        self.mass2 = fDCO["Mass_2"][()][self.DCOmask]

        # Stuff of data I dont need for integral
        # but I might be to laze to read in myself
        # and often use. Might turn it of for memory efficiency
        if self.lazyData:
            self.q = np.divide(self.mass2, self.mass1)
            boolq = self.mass2 > self.mass1
            self.q[boolq] = np.divide(self.mass1[boolq], self.mass2[boolq])
            self.mChirp = np.divide(
                (np.multiply(self.mass2, self.mass1) ** (3.0 / 5.0)),
                (np.add(self.mass2, self.mass1) ** (1.0 / 5.0)),
            )
            self.Hubble = fDCO["Merges_Hubble_Time"][...].squeeze()[self.DCOmask]

        Data.close()

    def getMassFormedAtEachZ(self, Mlower, Mupper, binaryFraction):
        """Set attributes related to mass sampling and calculates the total
        star-forming mass at each metallicity corresponding to the mass drawn
        and actually evolved by COMPAS.

        Args:
            Mlower ([float]): Minimum mass when drawing from Kroupa IMF.
            Mupper ([float]): Maximum mass when drawing from Kroupa IMF.
            binaryFraction ([float]): Binary fraction.
        """
        self.Mlower = Mlower
        self.Mupper = Mupper
        self.binaryFraction = binaryFraction
        _, self.massFormedAtEachZ = MPZ.getMassFormedAtEachZ(
            path=self.path,
            Mlower=self.Mlower,
            Mupper=self.Mupper,
            binaryFraction=self.binaryFraction,
        )
