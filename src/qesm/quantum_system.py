### qiskit and quantum chemistry modules ###
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp ,SparseLabelOp, PolynomialTensor, tensor_ordering
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import EvolvedOperatorAnsatz

######### qesm modules + others #########
from pyscf import gto, ao2mo
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from typing import Any, Dict, Union

class QuantumSystem:
    """
    Base class defining a quantum system for calculating spectral moments and one/two particle density matrices.
    Here we will define the system using PySCF's RHF mean-field object, and then use Qiskit to define
    the qubit hamiltoninan and ansatz for VQE calculations. 

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF RHF mean-field object.
    basis : {'MO', 'AO'}, optional
        Basis for integrals ('MO' for molecular orbital, 'AO' for atomic orbital). Default is 'MO'.
    reps : int, optional
        Number of repetitions for UCC ansatz. Default is 1.
    ansatz : QuantumCircuit or EvolvedOperatorAnsatz or None, optional
        Custom ansatz. If None, uses default UCCSD.
    """

    def __init__(
        self,
        mf: Any,
        basis: str = 'MO',
        reps: int = 1,
        ansatz: Union[QuantumCircuit, EvolvedOperatorAnsatz, None] = None
    ):
        # Validate input
        if basis not in ('MO', 'AO'):
            raise ValueError("basis must be 'MO' or 'AO'")

        self.mf = mf
        self.mol = mf.mol
        self.reps = reps

        self.mo_energy = self.mf.mo_energy
        self.mo_coeff = self.mf.mo_coeff
        self.mo_occ = self.mf.mo_occ

        self.num_orbitals = self.mo_coeff.shape[0]
        self.num_spin_orbitals = self.num_orbitals * 2
        self.num_particles = self.mol.nelec
        self.nuclear_repulsion_energy = gto.mole.energy_nuc(self.mol)

        # One/two body integrals
        if basis == 'MO':
            self.one_body = self.mo_coeff.T @ mf.get_hcore() @ self.mo_coeff
            eri = ao2mo.kernel(self.mf._eri, (self.mo_coeff,)*4, compact=False)
            self.two_body = eri.reshape((self.mo_coeff.shape[-1],) * 4)
        elif basis == 'AO':
            self.one_body = mf.get_hcore()
            self.two_body = mf._eri

        # Qubit Hamiltonian
        self.integrals = ElectronicIntegrals.from_raw_integrals(
            h1_a=self.one_body, h1_b=self.one_body,
            h2_aa=self.two_body, h2_bb=self.two_body, h2_ba=self.two_body, auto_index_order=True
        )
        self.h_elec = ElectronicEnergy(self.integrals, constants={'nuclear_repulsion_energy': self.nuclear_repulsion_energy}).second_q_op()
        self.mapper = JordanWignerMapper()
        self.qubit_ham = self.mapper.map(self.h_elec)
        self.n_qubits = self.qubit_ham.num_qubits

        # Ansatz
        if ansatz is None:
            initial_state = HartreeFock(
                num_spatial_orbitals=self.num_orbitals,
                num_particles=self.num_particles,
                qubit_mapper=self.mapper,
            )
            self.ansatz = UCC(
                num_spatial_orbitals=self.num_orbitals,
                num_particles=self.num_particles,
                excitations='sd',
                initial_state=initial_state,
                qubit_mapper=self.mapper,
                generalized=True,
                reps=self.reps,
            )
        else:
            self.ansatz = ansatz