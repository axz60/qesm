# PySCF, Vayesta and Dyson imports
from pyscf import fci
from vayesta.lattmod import Hubbard1D, LatticeRHF
from dyson import MBLGF, MixedMBLGF, util, NullLogger


# General libraries
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore")
from typing import Any, Dict, Union


'''
A utilities file to calculate system properties with spectral moments methods 
'''

def galitski_migdal_energy(
    METHOD: object,
    maxiter: int = 100,
    optimizer: str = "slsqp",
    for_double_occupation: bool = False,
    initial_point: Union[list, np.ndarray, None] = None
    ) -> float:
    """
    Calculate the Galitski-Migdal energy can be used to estimate the ground state energy of a system
    using zeroth and first spectral moments as specified in the paper by 
    Fertita et al. (https://arxiv.org/pdf/1904.08019). Note that Fertita et al. define the Galitski-Migdal 
    energy in GHF and UHF, for RHF, the factor of 0.5 is not applied.

    Parameters
    ----------
    METHOD : object
        The method object containing the system information.
    maxiter : int, optional
        Maximum number of iterations for VQE optimization, by default 100.
    optimizer : str, optional
        Optimizer to use for the calculation, by default "slsqp".
    for_double_occupation : bool, optional
        If True, the method is used to calculate double occupation, by default False.
    initial_point : Union[list, np.ndarray, None], optional
        Initial parameters for the ansatz. If None, randomly generated.

    Returns
    -------
    float
        The Galitski-Migdal energy value.
    """

    hole_moments = METHOD.hole_moments(nmom = 1, maxiter = maxiter, optimizer = optimizer, initial_point = initial_point)['moments']

    energy = np.einsum('ij,ji->', hole_moments[0], METHOD.one_body) + np.einsum('ij->', hole_moments[1])

    if for_double_occupation:
        outputs = {
            "energy": energy,
            "density matrix": hole_moments[0],
        }
        return outputs
    else:
        return energy

def double_occupation(
    METHOD: object,
    maxiter: int = 100,
    optimizer: str = "slsqp",
    gm_energy: bool = False,
    initial_point: Union[list, np.ndarray, None] = None
    )-> float:
    """
    Calculate the double occupation of a system using VQE or 
    spectral moments methods via Galitski-Migdal energy.

    Parameters
    ----------
    METHOD : object
        The method object containing the system information.
    maxiter : int, optional
        Maximum number of iterations for optimization, by default 100.
    optimizer : str, optional
        Optimizer to use for the calculation, by default "slsqp".
    gm : bool, optional
        If True, use Galitski-Migdal energy method, by default False.
    initial_point : Union[list, np.ndarray, None], optional
        Initial parameters for the ansatz. If None, randomly generated.

    Returns
    -------
    float
        The double occupation value.

    Notes
    -----
    Currently, this function supports only supports Hubbard models with a single Hubbard U.
    """
    if not gm_energy:
        # Use VQE to get the double occupation
        dm_obj = METHOD.hole_moments(nmom=0, maxiter=maxiter, optimizer=optimizer, initial_point=initial_point)
        dm = dm_obj['density matrix'] # dm is a 2D array
        energy = dm_obj['ground state energy']

        double_occ = 2*(energy - np.einsum('ij,ji->', dm, METHOD.one_body) - METHOD.nuclear_repulsion_energy) / METHOD.mol.hubbard_u

        return double_occ
    
    elif gm_energy:
        # Use Galitski-Migdal energy method to get the double occupation
        outputs = galitski_migdal_energy(METHOD, maxiter=maxiter, optimizer=optimizer, for_double_occupation=True, initial_point=initial_point)
        density_matrix = outputs['density matrix']
        energy = outputs['energy']

        double_occ = 2*(energy - np.einsum('ij,ji->', density_matrix, METHOD.one_body) - METHOD.nuclear_repulsion_energy) / METHOD.mol.hubbard_u

        return double_occ

def spectral_function(
    METHOD: object,
    omega_grid: np.ndarray,
    nmom: int = 1,
    maxiter: int = 100,
    optimizer: str = "slsqp",
    eta: float = 0.01,
    initial_point: Union[list, np.ndarray, None] = None
    )-> dict:
    """
    Calculate the spectral function of a system using spectral moments methods.

    Parameters
    ----------
    METHOD : object
        The method object containing the system information.
    nmom : int, optional
        Number of moments to calculate, by default 1.
    maxiter : int, optional
        Maximum number of iterations for optimization, by default 100.
    optimizer : str, optional
        Optimizer to use for the calculation, by default "slsqp".
    omega_grid : np.ndarray
        The grid of frequencies for which to calculate the spectral function.
    eta : float, optional
        Broadening parameter for the spectral function, by default 0.01.
    initial_point : Union[list, np.ndarray, None], optional
        Initial parameters for the ansatz. If None, randomly generated.

    Returns
    -------
    dict
        A dictionary containing the spectral function and the omega grid.

    """
    hole_moments = METHOD.hole_moments(nmom=nmom, maxiter=maxiter, optimizer=optimizer, initial_point=initial_point)['moments']
    particle_moments = METHOD.particle_moments(nmom=nmom, maxiter=maxiter, optimizer=optimizer, initial_point=initial_point)['moments']

    hole_mblgf = MBLGF(hole_moments,log =NullLogger())
    particle_mblgf = MBLGF(particle_moments, log=NullLogger())
    solver = MixedMBLGF(hole_mblgf, particle_mblgf)
    solver.kernel() 
    poles,dyson_orbitals = solver.get_dyson_orbitals()
    spectral_function = util.build_spectral_function(poles, dyson_orbitals,omega_grid, eta=0.01)

    outputs = {'spectral function': spectral_function, 'grid': omega_grid}

    return outputs

