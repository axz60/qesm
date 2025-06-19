### qiskit and quantum chemistry modules ###
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp ,SparseLabelOp, PolynomialTensor
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
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from typing import Any, Dict, Union


class DirectMethod(QuantumSystem):
    """
    Direct calculation of spectral moments and related quantities using Qiskit and PySCF.

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

    def hole_moments(
        self,
        nmom: int,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        initial_point: Union[list, np.ndarray, None] = None
        ) -> dict:
        """
        Compute hole moments of the system.

        Parameters
        ----------
        nmom : int
            Number of moments to compute.
        maxiter : int, optional
            Maximum number of iterations for VQE. Default is 100.
        optimizer : str, optional
            Optimizer to use for VQE. Default is 'slsqp'.
        initial_point : list, np.ndarray, or None, optional
            Initial parameters for the ansatz. If None, randomly generated.

        Returns
        -------
        dict
            Dictionary containing computed moments and related data (NotImplemented).

        """
        moment_list = []
        temp_moments = []
        plus_sq = [] 
        plus_pauli = []
        minus_sq = []
        minus_pauli = []

        off_diag_moment_list = []
        off_diag_temp_moments = []

        moment_matrix = []
        temp_matrix = np.zeros((self.num_orbitals,self.num_orbitals))

        vqe = self.aer_vqe_hamiltonian(maxiter=maxiter,optimizer=optimizer, initial_point=initial_point)
        optimal_circ = vqe['opt_circuit']
        E = vqe['opt_energy']
        data = {}
        data['']= E
        tensor = PolynomialTensor(data)
        E_term = FermionicOp.from_polynomial_tensor(tensor)
        power = -self.h_elec + E_term
        qubit_power = SparsePauliOp.from_operator(self.mapper.map(power))

        for i in range(self.num_orbitals): 

            plus_sq.append(FermionicOp({ '+_{}'.format(i) : 1}, num_spin_orbitals= self.num_spin_orbitals))
            plus_pauli.append(SparsePauliOp.from_operator(self.mapper.map(plus_sq[i])))
            minus_sq.append(FermionicOp({ '-_{}'.format(i) : 1}, num_spin_orbitals= self.num_spin_orbitals))
            minus_pauli.append(SparsePauliOp.from_operator(self.mapper.map(minus_sq[i])))

            temp_moments.append(self.expval(optimal_circ,plus_pauli[i] @ minus_pauli[i]))
            temp_matrix[i,i] = temp_moments[i].real

        for i in range(self.num_orbitals):
            for j in range(self.num_orbitals):
                if i != j:
                    off_diag_temp_moments.append( self.expval(optimal_circ,(plus_pauli[i]) @ ( minus_pauli[j])))
                    temp_matrix[i,j] = off_diag_temp_moments[-1].real

        # If nmom is 0, return the density matrix and ground state energy
        if nmom == 0:
            output = {'density matrix':temp_matrix, 'ground state energy':E}
            return output

        moment_list.append(temp_moments)
        off_diag_moment_list.append(off_diag_temp_moments)
        moment_matrix.append(temp_matrix)

        def get_moments(nmom): 

            new_moment_list = []
            off_diag_new_moment_list = []
            matrix = np.zeros((self.num_orbitals,self.num_orbitals))
            ting = Operator(np.linalg.matrix_power(Operator(qubit_power).to_matrix(),nmom))

            for i in range(self.num_orbitals):

                qubit_op_ii = Operator(plus_pauli[i]) @ ting @ Operator(minus_pauli[i])
                qubit_op_ii = SparsePauliOp.from_operator(qubit_op_ii)
                new_moment_list.append(self.expval(optimal_circ,qubit_op_ii))
                matrix[i,i] = new_moment_list[i]

                for j in range(self.num_orbitals):

                    if i != j:

                        qubit_op_ij = Operator(plus_pauli[i]) @ ting @ Operator(minus_pauli[j])
                        qubit_op_ij = SparsePauliOp.from_operator(qubit_op_ij)
                        off_diag_new_moment_list.append(self.expval(optimal_circ,qubit_op_ij))
                        matrix[i,j] = off_diag_new_moment_list[-1].real
                
            outputs = {'new moments':new_moment_list,'off diag moments':off_diag_new_moment_list,'matrix':matrix}

            return outputs
        
        for i in range(1,nmom+1):

            new_moments = get_moments(i)

            moment_list.append(new_moments['new moments'])
            off_diag_moment_list.append(new_moments['off diag moments'])
            moment_matrix.append(new_moments['matrix'])

        moments = np.concatenate(moment_matrix,axis = 0).reshape(nmom+1,self.num_orbitals,self.num_orbitals)

        output = {'moments':moments}

        return output

    def particle_moments(       
        self,
        nmom: int,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        initial_point: Union[list, np.ndarray, None] = None
        ) -> list:
        """
        Compute particle moments of the system.

        Parameters
        ----------
        nmom : int
            Number of moments to compute.
        maxiter : int, optional
            Maximum number of iterations for VQE. Default is 100.
        optimizer : str, optional
            Optimizer to use for VQE. Default is 'slsqp'.

        Returns
        -------
        dict
            Dictionary containing computed moments and related data.
        """
        moment_list = []
        temp_moments = []
        plus_sq = [] 
        plus_pauli = []
        minus_sq = []
        minus_pauli = []

        off_diag_moment_list = []
        off_diag_temp_moments = []

        moment_matrix = []
        temp_matrix = np.zeros((self.num_orbitals,self.num_orbitals))

        vqe = self.aer_vqe_hamiltonian(maxiter=maxiter,optimizer=optimizer, initial_point=initial_point)
        optimal_circ = vqe['opt_circuit']
        E = vqe['opt_energy']
        data = {}
        data['']= E 
        tensor = PolynomialTensor(data)
        E_term = FermionicOp.from_polynomial_tensor(tensor)
        power = self.h_elec - E_term
        qubit_power = SparsePauliOp.from_operator(self.mapper.map(power))

        for i in range(self.num_orbitals): 

            plus_sq.append(FermionicOp({ '+_{}'.format(i) : 1}, num_spin_orbitals= self.num_spin_orbitals))
            plus_pauli.append(SparsePauliOp.from_operator(self.mapper.map(plus_sq[i])))
            minus_sq.append(FermionicOp({ '-_{}'.format(i) : 1}, num_spin_orbitals= self.num_spin_orbitals))
            minus_pauli.append(SparsePauliOp.from_operator(self.mapper.map(minus_sq[i])))

            temp_moments.append(self.expval(optimal_circ,minus_pauli[i] @ plus_pauli[i]))
            temp_matrix[i,i] = temp_moments[i].real

        for i in range(self.num_orbitals):
            for j in range(self.num_orbitals):
                    
                    if i != j:
                        off_diag_temp_moments.append(self.expval(optimal_circ,(minus_pauli[i] ) @ (plus_pauli[j] )))
                        temp_matrix[i,j] = off_diag_temp_moments[-1].real

        # If nmom is 0, return the density matrix and ground state energy                              
        if nmom == 0:
            output = {'density matrix':temp_matrix, 'ground state energy':E}
            return output

        moment_list.append(temp_moments)
        off_diag_moment_list.append(off_diag_temp_moments)
        moment_matrix.append(temp_matrix)

        def get_moments(nmom): 

            new_moment_list = []
            off_diag_new_moment_list = []
            matrix = np.zeros((self.num_orbitals,self.num_orbitals))
            ting = Operator(np.linalg.matrix_power(Operator(qubit_power).to_matrix(),nmom))


            for i in range(self.num_orbitals):

                qubit_op_ii = Operator(minus_pauli[i]) @ ting @ Operator(plus_pauli[i])
                qubit_op_ii = SparsePauliOp.from_operator(qubit_op_ii)
                new_moment_list.append(self.expval(optimal_circ,qubit_op_ii))
                matrix[i,i] = new_moment_list[i]

                for j in range(self.num_orbitals):

                    if i != j:

                        qubit_op_ij = Operator(minus_pauli[i]) @ ting @ Operator(plus_pauli[j])
                        qubit_op_ij = SparsePauliOp.from_operator(qubit_op_ij)
                        off_diag_new_moment_list.append(self.expval(optimal_circ,qubit_op_ij))
                        matrix[i,j] = off_diag_new_moment_list[-1].real
   
            outputs = {'new moments':new_moment_list,'off diag moments':off_diag_new_moment_list,'matrix':matrix}

            return outputs
        
        for i in range(1,nmom+1):

            new_moments = get_moments(i)
            moment_list.append(new_moments['new moments'])
            off_diag_moment_list.append(new_moments['off diag moments'])
            moment_matrix.append(new_moments['matrix'])

        moments = np.concatenate(moment_matrix,axis = 0).reshape(nmom+1,self.num_orbitals,self.num_orbitals)

        output = {'moments':moments}

        return output      

    def get_galitski_migdal_energy(
        self,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        initial_point: Union[list, np.ndarray, None] = None
        ) -> float:
        """
        Calculate the Galitski-Migdal energy of the system.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations for VQE. Default is 100.
        optimizer : str, optional
            Optimizer to use for VQE. Default is 'slsqp'.
        initial_point : Union[list, np.ndarray, None], optional
            Initial parameters for the ansatz. If None, randomly generated.

        Returns
        -------
        float
            The Galitski-Migdal energy value.
        """
        energy = galitski_migdal_energy(METHOD=self, maxiter=maxiter, optimizer=optimizer, for_double_occupation=False, initial_point=initial_point)

        return energy

    def get_double_occupation(
        self,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        gm_energy: bool = False
        ) -> float:
        """
        Calculate the double occupation of the system using VQE or spectral moments methods.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations for optimization. Default is 100.
        optimizer : str, optional
            Optimizer to use for the calculation. Default is 'slsqp'.
        gm_energy : bool, optional
            If True, use Galitski-Migdal energy method. Default is False.

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
            d_o = double_occupation(METHOD=self, maxiter=maxiter, optimizer=optimizer, initial_point=None, gm_energy=False)
            return d_o
        else:
            # Use Galitski-Migdal energy method
            d_o = double_occupation(METHOD=self, maxiter=maxiter, optimizer=optimizer, gm_energy=gm_energy, initial_point=None)
            return d_o

    def get_spectral_function(
        self,
        omega_grid: np.ndarray,
        nmom: int = 1,
        maxiter: int = 100,
        optimizer: str = 'slsqp',
        eta: float = 0.01,
        initial_point: Union[list, np.ndarray, None] = None
        ) -> dict:
        """
        Calculate the spectral function of the system using spectral moments.

        Parameters
        ----------
        omega_grid : np.ndarray
            Frequency grid for the spectral function.
        nmom : int, optional
            Number of moments to compute. Default is 1.
        maxiter : int, optional
            Maximum number of iterations for optimization. Default is 100.
        optimizer : str, optional
            Optimizer to use for the calculation. Default is 'slsqp'.
        eta : float, optional
            Broadening parameter for the spectral function. Default is 0.1.
        initial_point : Union[list, np.ndarray, None], optional
            Initial parameters for the ansatz. If None, randomly generated.
        
        Returns
        -------
        dict
            Dictionary containing the spectral function values and frequencies.
        """
        spectral_func = spectral_function(
            METHOD=self,
            omega_grid=omega_grid,
            nmom=nmom,
            maxiter=maxiter,
            optimizer=optimizer,
            eta=eta,
            initial_point=initial_point
        )

        return spectral_func

    

