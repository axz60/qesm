# PySCF, Vayesta and Dyson imports 
from pyscf import fci,gto
from vayesta.lattmod import Hubbard1D, LatticeRHF

# General libraries
import numpy as np
from qesm.aer_rdm import *
from qesm.aer_util import primatives
import pytest
import warnings
warnings.filterwarnings("ignore")

def test_rdms():
    # Build simple 2-site Hubbard model (as before)
    nmom = 1
    omega_grid = np.linspace(-10, 10, 1000)
    u_hub = 3
    nsite = 2
    mol_hub = Hubbard1D(
        nsite=nsite,
        nelectron=nsite,
        hubbard_u=u_hub,
        verbose=0,
    )
    mf_hub = LatticeRHF(mol_hub)
    mf_hub.kernel()
    myci = fci.FCI(mf_hub, mf_hub.mo_coeff)
    fci_energy, fci_vecs = myci.kernel()

    rdm1 = myci.make_rdm1(fci_vecs, mf_hub.mo_coeff.shape[1],mol_hub.nelec)
    rdm2 = myci.make_rdm2(fci_vecs, mf_hub.mo_coeff.shape[1],mol_hub.nelec)

    fci_double_occupation = 2*(fci_energy - np.einsum('ij,ji->', rdm1, mf_hub.mo_coeff.T @ mf_hub.get_hcore() @ mf_hub.mo_coeff)) / mol_hub.hubbard_u 

    # build RDM object
    prims = primatives(shots=100000)
    sv_estimator = prims['sv estimator']
    estimator = prims['estimator']

    rdm_obj = RDMs(
        mf=mf_hub,
        vqe_estimator=sv_estimator,
        expectation_estimator=estimator,
    )

    rdm_energy = rdm_obj.get_energy() 
    rdm_double_occupation_using_rdms = rdm_obj.get_double_occupation(from_rdms=True)
    rdm_double_occupation_using_using_vqe = rdm_obj.get_double_occupation(from_rdms=False)

    # Check that the RDM energy is close to the FCI energy
    assert np.isclose(rdm_energy, fci_energy, atol=1e-2), f"RDM energy {rdm_energy} does not match FCI energy {fci_energy}"
    # Check that the double occupation from RDM is close to the FCI double occupation
    assert np.isclose(rdm_double_occupation_using_rdms, fci_double_occupation, atol=1e-2), f"RDM double occupation {rdm_double_occupation_using_rdms} does not match FCI double occupation {fci_double_occupation}"
    # Check that the double occupation from VQE is close to the FCI double occupation
    assert np.isclose(rdm_double_occupation_using_using_vqe, fci_double_occupation, atol=1e-2), f"VQE double occupation {rdm_double_occupation_using_using_vqe} does not match FCI double occupation {fci_double_occupation}"


