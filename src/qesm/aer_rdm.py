### qiskit and quantum chemistry modules ###
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp ,SparseLabelOp, PolynomialTensor, tensor_ordering
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import EvolvedOperatorAnsatz

### qiskit measurement modules ###
from qiskit.primitives import BaseEstimatorV1, BaseEstimatorV2
from qiskit_aer import AerSimulator

######### qesm modules + others #########
from .aer_util import * 
from .system_properties import *
from .quantum_system import QuantumSystem
from pyscf import gto, ao2mo
import itertools
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from typing import Any, Dict, Union

class RDMs(QuantumSystem):
    """
    Constructing one and two electron reduced density matrices (RDMs) using PySCF and Qiskit.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF RHF mean-field object.
    vqe_estimator : BaseEstimatorV1
        Qiskit estimator primitive for VQE.
    expectation_estimator : BaseEstimatorV1
        Estimator for measuring expectation values.
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
        vqe_estimator: BaseEstimatorV1,
        expectation_estimator: BaseEstimatorV1,
        basis: str = 'MO',
        reps: int = 1,
        ansatz: Union[QuantumCircuit, EvolvedOperatorAnsatz, None] = None
        ):
        super().__init__(mf=mf, basis=basis, reps=reps, ansatz=ansatz)

        # Validate estimator types
        if not isinstance(vqe_estimator, BaseEstimatorV1):
            raise TypeError("vqe_estimator must be a BaseEstimatorV1 instance. Estimator V2 is not implemented.")

        self.sv_estimator = vqe_estimator
        self.stoc_estimator = expectation_estimator

    def aer_vqe_hamiltonian(
        self,
        optimizer: str = 'slsqp',
        maxiter: int = 100,
        initial_point: Union[list, np.ndarray, None] = None
        ) -> Dict[str, Any]:
        """
        Run VQE on the system Hamiltonian using the provided ansatz and estimator.

        Parameters
        ----------
        optimizer : str, optional
            Optimizer to use ('spsa', 'slsqp', 'l-bfgs-b', 'cobyla'). Default is 'slsqp'.
        maxiter : int, optional
            Maximum number of iterations for the optimizer. Default is 100.
        initial_point : list, np.ndarray, or None, optional
            Initial parameters for the ansatz. If None, randomly generated.

        Returns
        -------
        dict
            Result dictionary with keys: 'opt_params', 'opt_circuit', 'opt_objective', 'opt_energy'.
        """
        if initial_point is None:
            initial_point = self.ansatz.num_parameters*[np.random.uniform(0.0, 10.0)]

        result = run_vqe(
            estimator=self.sv_estimator,
            ansatz=self.ansatz,
            qubit_hamiltonian=self.qubit_ham,
            optimizer=optimizer,
            maxiter=maxiter,
            initial_point=initial_point,
            nuclear_repulsion_energy=self.nuclear_repulsion_energy
        )
        
        return result

    def expval(
        self,
        circuit: Union[QuantumCircuit, EvolvedOperatorAnsatz],
        qubit_hamiltonian: SparsePauliOp,
        ) -> float:
        """
        Compute the expectation value of an operator using the provided circuit.

        Parameters
        ----------
        circuit : QuantumCircuit or EvolvedOperatorAnsatz
            The quantum circuit to use for the measurement.
        operator : SparsePauliOp
            The operator for which to compute the expectation value.

        Returns
        -------
        float
            The expectation value of the operator.
        """
        if not isinstance(qubit_hamiltonian, SparsePauliOp):
            raise TypeError("Pauli operator must be type SparsePauliOp")
        
        exp_value = aer_expectation_value(estimator=self.stoc_estimator, qubit_hamiltonian = qubit_hamiltonian, circuit=circuit)
        
        return exp_value

    def make_1rdm(
        self,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        initial_point: Union[list, np.ndarray, None] = None,
        ) -> Dict[str, Any]:
        """
        Calculate one particle reduced density matrix (1-RDM) using VQE.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations for the optimizer. Default is 100.
        optimizer : str, optional
            Optimizer to use ('spsa', 'slsqp', 'l-bfgs-b', 'cobyla'). Default is 'slsqp'.
        initial_point : list, np.ndarray, or None, optional
            Initial parameters for the ansatz. If None, randomly generated.

        Returns
        -------
        dict
            Result dictionary with keys: 'one_rdm', 'energy'.
        """
        vqe = self.aer_vqe_hamiltonian(
            optimizer=optimizer,
            maxiter=maxiter,
            initial_point=initial_point
        )
        optimal_circ = vqe['opt_circuit']               
        # initialising the reduced density matricies arrays
        one_rdm = np.zeros(tuple([self.num_orbitals,]*2))
        tmp_h1 = np.zeros(self.one_body.shape)
        tmp_h2 = np.zeros(self.two_body.shape)

        for i , j in itertools.product(range(self.num_orbitals), repeat=2):

            tmp_h1[i,j] = 1
                                
            tmp_integrals = ElectronicIntegrals.from_raw_integrals(h1_a=tmp_h1, h1_b= tmp_h1, h2_aa=tmp_h2, h2_bb=tmp_h2 ,h2_ba=tmp_h2,auto_index_order = False)
            tmp_operator = ElectronicEnergy(tmp_integrals).second_q_op()      
            tmp_qubit_ham = self.mapper.map(tmp_operator)
            
            ene_tmp = self.expval(optimal_circ,tmp_qubit_ham) 
            one_rdm[i,j] = ene_tmp 
            tmp_h1[i,j] = 0.

        output = {
            'one_rdm': one_rdm,
            'energy': vqe['opt_energy']
        }

        return output

    def make_2rdm(
        self,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        initial_point: Union[list, np.ndarray, None] = None,
        ) -> Dict[str, Any]:
        """
        Calculate two particle reduced density matrix (1-RDM) using VQE.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations for the optimizer. Default is 100.
        optimizer : str, optional
            Optimizer to use ('spsa', 'slsqp', 'l-bfgs-b', 'cobyla'). Default is 'slsqp'.
        initial_point : list, np.ndarray, or None, optional
            Initial parameters for the ansatz. If None, randomly generated.

        Returns
        -------
        dict
            Result dictionary with keys: 'one_rdm', 'energy'.
        """
        vqe = self.aer_vqe_hamiltonian(
            optimizer=optimizer,
            maxiter=maxiter,
            initial_point=initial_point
        )
        optimal_circ = vqe['opt_circuit']               
        # initialising the reduced density matricies arrays
        tmp_h1 = np.zeros(self.one_body.shape)
        two_rdm = np.zeros(tuple([self.num_orbitals]*4))
        tmp_h2 = np.zeros(self.two_body.shape)

        for i , j , k , l in itertools.product(range(self.num_orbitals), repeat=4):
            tmp_h2[i,j,k,l] = 1
               
            tmp_integrals = ElectronicIntegrals.from_raw_integrals(h1_a=tmp_h1, h1_b= tmp_h1, h2_aa=tmp_h2, h2_bb=tmp_h2 ,h2_ba=tmp_h2,auto_index_order = False) 
            tmp_operator = ElectronicEnergy(tmp_integrals).second_q_op()      
            tmp_qubit_ham = self.mapper.map(tmp_operator)
       
            ene_tmp = self.expval(optimal_circ,tmp_qubit_ham) 
            two_rdm[i,j,k,l] = ene_tmp
            tmp_h2[i,j,k,l] = 0.
        
        output = {
            'two_rdm': 2*two_rdm,
            'energy': vqe['opt_energy']
        }

        return output

    def get_energy(
        self,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        initial_point: Union[list, np.ndarray, None] = None,
        ) -> float:
        """
        Calculate energy with density matrix elements using VQE.
        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations for the optimizer. Default is 100.
        optimizer : str, optional
            Optimizer to use ('spsa', 'slsqp', 'l-bfgs-b', 'cobyla'). Default is 'slsqp'.
        initial_point : list, np.ndarray, or None, optional
            Initial parameters for the ansatz. If None, randomly generated.
        
        Returns
        -------
        float
            The calculated energy.
        """
        one_rdm = self.make_1rdm(
            maxiter=maxiter,
            optimizer=optimizer,
            initial_point=initial_point
        )['one_rdm']

        two_rdm = self.make_2rdm(
            maxiter=maxiter,
            optimizer=optimizer,
            initial_point=initial_point
        )['two_rdm']

        energy = np.einsum('ij,ji->', one_rdm, self.one_body) + 0.5*np.einsum('ijkl,ijkl->', two_rdm, self.two_body) + self.nuclear_repulsion_energy
        return energy

    def get_double_occupation(
        self,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        initial_point: Union[list, np.ndarray, None] = None,
        from_rdms: bool = False
        ) -> float:
        """
        Calculate the double occupation of a system using VQE or spectral moments methods.
        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations for optimization, by default 100.
        optimizer : str, optional
            Optimizer to use for the calculation, by default "slsqp".
        initial_point : Union[list, np.ndarray, None], optional
            Initial parameters for the ansatz. If None, randomly generated.
        from_rdms : bool, optional
            If True, use the reduced density matrices to calculate double occupation.
        Returns
        -------
        float
            The double occupation value.

        Notes
        -----
        Currently, this function supports only supports Hubbard models with a single Hubbard U.
        """

        one_rdm_obj = self.make_1rdm(
            maxiter=maxiter,
            optimizer=optimizer,
            initial_point=initial_point
        )
        energy = one_rdm_obj['energy']
        one_rdm = one_rdm_obj['one_rdm']

        if from_rdms:
            energy_from_rdms = self.get_energy(
                maxiter=maxiter,
                optimizer=optimizer,
                initial_point=initial_point
            )
            double_occ = 2*(energy_from_rdms - np.einsum('ij,ji ->', one_rdm,self.one_body) - self.nuclear_repulsion_energy) / self.mol.hubbard_u
            return double_occ

        else:
            double_occ = 2 * (energy - np.einsum('ij,ji->', one_rdm, self.one_body) - self.nuclear_repulsion_energy) / self.mol.hubbard_u
            return double_occ

