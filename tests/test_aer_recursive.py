# PySCF, Vayesta and Dyson imports 
from pyscf import fci,gto
from vayesta.lattmod import Hubbard1D, LatticeRHF
from dyson import MBLGF, MixedMBLGF, FCI, Lehmann, util, NullLogger
from dyson.util import greens_function_galitskii_migdal

# General and Qesm imports
import numpy as np
from qesm.aer_recursive import *
from qesm.aer_util import primatives
import pytest
import warnings
warnings.filterwarnings("ignore")

### Test hole moments shape and type ###
def test_hole_moments():
    # Build simple 2-site Hubbard model (as before)
    nmom = 2
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

    # Use statevector estimators (no shot noise)
    prims = primatives(shots=100000)
    sv_estimator = prims['sv estimator']
    simulator = prims['simulator']

    # fci moments
    expr = FCI["1h"](mf_hub, mo_energy=mf_hub.mo_energy, mo_coeff=mf_hub.mo_coeff, mo_occ=mf_hub.mo_occ)
    th_fci = expr.build_gf_moments(nmom+1)

    # Build RecursiveMethod object
    dm = RecursiveMethod(
        mf=mf_hub,
        vqe_estimator=sv_estimator,
        simulator=simulator,
    )

    # Compute hole moments for 2 moments (should work for nmom=2)
    output = dm.hole_moments(nmom=nmom, maxiter=40, optimizer='slsqp')

    # # compare with FCI moments
    moments = output['moments']

    assert np.isclose(np.trace(moments[0]), np.trace(th_fci[0]), atol=1e-1), f"zero-th moment mismatch: {np.trace(moments[0])} vs {np.trace(th_fci[0])}"
    assert np.isclose(np.trace(moments[1]), np.trace(th_fci[1]), atol=1e-1), f"first moment mismatch: {np.trace(moments[1])} vs {np.trace(th_fci[1])}"
    assert np.isclose(np.trace(moments[2]), np.trace(th_fci[2]), atol=1e-1), f"second moment mismatch: {np.trace(moments[2])} vs {np.trace(th_fci[2])}"

### Test hole moments shape and type ###

### Test particle moments shape and type ###
def test_particle_moments():
    # Build simple 2-site Hubbard model (as before)
    nmom = 2
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

    # initialize primatives
    prims = primatives(shots=100000)
    sv_estimator = prims['sv estimator']
    simulator = prims['simulator']

    # fci moments
    expr = FCI["1p"](mf_hub, mo_energy=mf_hub.mo_energy, mo_coeff=mf_hub.mo_coeff, mo_occ=mf_hub.mo_occ)
    tp_fci = expr.build_gf_moments(nmom+1)

    # Build RecursiveMethod object
    dm = RecursiveMethod(
        mf=mf_hub,
        vqe_estimator=sv_estimator,
        simulator=simulator,
    )

    # Compute hole moments for 2 moments (should work for nmom=2)
    output = dm.particle_moments(nmom=nmom, maxiter=40, optimizer='slsqp')

    # # compare with FCI moments
    moments = output['moments']

    assert np.isclose(np.trace(moments[0]), np.trace(tp_fci[0]), atol=1e-1), f"zero-th moment mismatch: {np.trace(moments[0])} vs {np.trace(tp_fci[0])}"
    assert np.isclose(np.trace(moments[1]), np.trace(tp_fci[1]), atol=1e-1), f"first moment mismatch: {np.trace(moments[1])} vs {np.trace(tp_fci[1])}"
    assert np.isclose(np.trace(moments[2]), np.trace(tp_fci[2]), atol=1e-1), f"second moment mismatch: {np.trace(moments[2])} vs {np.trace(tp_fci[2])}"

### System properties tests ###
def test_system_properties():
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
    fci_energy,_ = myci.kernel()

    # Build FCI moments
    expr = FCI["1p"](mf_hub, mo_energy=mf_hub.mo_energy, mo_coeff=mf_hub.mo_coeff, mo_occ=mf_hub.mo_occ)
    tp_fci = expr.build_gf_moments(nmom+1)
    expr = FCI["1h"](mf_hub, mo_energy=mf_hub.mo_energy, mo_coeff=mf_hub.mo_coeff, mo_occ=mf_hub.mo_occ)
    th_fci = expr.build_gf_moments(nmom+1)

    # FCI GM energy
    e_fci = greens_function_galitskii_migdal(th_fci, mf_hub.mo_coeff.T @ mf_hub.get_hcore() @ mf_hub.mo_coeff, factor=1)

    # FCI double occupation via full system energy
    double_occ_fci = 2 * (fci_energy - np.einsum('ij,ji->', th_fci[0], mf_hub.mo_coeff.T @ mf_hub.get_hcore() @ mf_hub.mo_coeff )) / mol_hub.hubbard_u

    # FCI double occupation via galitski-migdal energy
    double_occ_fci_gm = 2 * (e_fci - np.einsum('ij,ji->', th_fci[0], mf_hub.mo_coeff.T @ mf_hub.get_hcore() @ mf_hub.mo_coeff )) / mol_hub.hubbard_u

    # FCI spectral function
    # solverh = MBLGF(th_fci, log=NullLogger())
    # solverp = MBLGF(tp_fci, log=NullLogger())
    # solver = MixedMBLGF(solverh, solverp)
    # solver.kernel()
    # E,V = solver.get_dyson_orbitals()
    # sf_fci = util.build_spectral_function(E, V, omega_grid, eta=0.1)

    # build RecursiveMethod object
    prims = primatives(shots=100000)
    sv_estimator = prims['sv estimator']
    simulator = prims['simulator']

    dm_obj = RecursiveMethod(
        mf=mf_hub,
        vqe_estimator=sv_estimator,
        simulator=simulator,
    )
    vqe_result = dm_obj.aer_vqe_hamiltonian(
        optimizer='slsqp',
        maxiter=100
    )
    e_vqe = vqe_result['opt_energy']

    # recMethod galitski-migdal energy
    e_rec = dm_obj.get_galitski_migdal_energy(maxiter=100, optimizer='slsqp')

    # recMethod double occupation via full system vqe energy
    double_occ_rec = dm_obj.get_double_occupation(maxiter=100, optimizer='slsqp',gm_energy=False)

    # recMethod double occupation via galitski-migdal energy
    double_occ_rec_gm = dm_obj.get_double_occupation(maxiter=100, optimizer='slsqp', gm_energy=True)

    # recMethod spectral function
    # spectral_function = dm_obj.get_spectral_function(
    #     omega_grid=omega_grid,
    #     nmom=nmom,
    #     maxiter=100,
    #     optimizer='slsqp',
    #     eta=0.1
    # )

    # Check Galitski-Migdal energy
    assert np.isclose(e_rec, e_fci, atol=1e-2), f"Galitski-Migdal energy mismatch: {e_rec:.6f} vs {e_fci:.6f}"
    # Check double occupation
    assert np.isclose(double_occ_rec, double_occ_fci, atol=1e-2), f"Double occupation mismatch: {double_occ_rec:.6f} vs {double_occ_fci:.6f}"
    # Check gm energies for fci
    assert np.allclose(e_fci,fci_energy, atol=1e-2), f"FCI energy mismatch: {e_fci:.6f} vs {fci_energy:.6f}"
    # check gm energies for rec method and fci
    assert np.allclose(e_rec, fci_energy, atol=1e-2), f"recMethod energy mismatch: {e_rec:.6f} vs {fci_energy:.6f}"
    # Check double occupation via GM energy
    assert np.isclose(double_occ_rec_gm, double_occ_fci_gm, atol=1e-2), f"Double occupation via GM energy mismatch: {double_occ_rec_gm:.6f} vs {double_occ_fci_gm:.6f}"
    # Check if rec method gm double occupation matches rec method double occupation without gm energy
    assert np.isclose(double_occ_rec, double_occ_rec_gm, atol=1e-2), f"recMethod double occupation mismatch: {double_occ_rec:.6f} vs {double_occ_rec_gm:.6f}"
    # Check if vqe energy matches rec gm energy
    assert np.isclose(e_vqe, e_rec, atol=1e-2), f"VQE energy mismatch: {e_vqe:.6f} vs {e_rec:.6f}"


