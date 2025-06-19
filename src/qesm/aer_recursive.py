### qiskit and quantum chemistry modules ###
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile
from qiskit.quantum_info import Statevector, SparsePauliOp, DensityMatrix
from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp, PolynomialTensor
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_nature.second_q.circuit.library import HartreeFock, PUCCSD

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

class RecursiveMethod(QuantumSystem):
    """
    Calculating spectral moments via the and related quantities via the recursive method using Qiskit and PySCF.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF RHF mean-field object.
    vqe_estimator : BaseEstimatorV1
        Qiskit estimator primitive for VQE.
    simulator : AerSimulator
        Qiskit Aer simulator instance to sample the quantum circuits classically.
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
        simulator: AerSimulator,
        basis: str = 'MO',
        reps: int = 1,
        ansatz: Union[QuantumCircuit, EvolvedOperatorAnsatz, None] = None
        ):

        super().__init__(mf=mf, basis=basis, reps=reps, ansatz=ansatz)

        # Validate estimator types
        if not isinstance(vqe_estimator, BaseEstimatorV1):
            raise TypeError("vqe_estimator must be a BaseEstimatorV1 instance. Estimator V2 is not implemented.")

        self.estimator = vqe_estimator
        self.simulator = simulator

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
            estimator=self.estimator,
            ansatz=self.ansatz,
            qubit_hamiltonian=self.qubit_ham,
            optimizer=optimizer,
            maxiter=maxiter,
            initial_point=initial_point,
            nuclear_repulsion_energy=self.nuclear_repulsion_energy
        )
        
        return result     

    def aer_vqe_general(
        self,
        initial_point: Union[list, np.ndarray],
        ansatz: Union[QuantumCircuit, EvolvedOperatorAnsatz],
        qubit_hamiltonian: SparsePauliOp,
        optimizer: str = 'slsqp',
        maxiter: int = 100,
        ) -> Dict[str, Any]:
        """
        Run VQE on a general Hamiltonian using the provided ansatz and estimator.

        Parameters
        ----------
        initial_point : list or np.ndarray
            Initial parameters for the ansatz.
        ansatz : QuantumCircuit or EvolvedOperatorAnsatz
            The ansatz to use for VQE.
        qubit_hamiltonian : SparsePauliOp
            The Hamiltonian in qubit representation.
        optimizer : str, optional
            Optimizer to use ('spsa', 'slsqp', 'l-bfgs-b', 'cobyla'). Default is 'slsqp'.
        maxiter : int, optional
            Maximum number of iterations for the optimizer. Default is 100.

        Returns
        -------
        dict
            Result dictionary with keys: 'opt_params', 'opt_circuit', 'opt_objective', 'opt_energy'.
        """
        result = run_vqe(
            estimator=self.estimator,
            ansatz=ansatz,
            qubit_hamiltonian=qubit_hamiltonian,
            optimizer=optimizer,
            maxiter=maxiter,
            initial_point=initial_point,
            nuclear_repulsion_energy=self.nuclear_repulsion_energy
        )
        return result   

    def get_ovlp(
        self,
        adjoint_state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
        state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
        ) -> complex:
        """
        Calculate the overlap between two quantum states 

        Parameters
        ----------
        adjoint_state : QuantumCircuit or EvolvedOperatorAnsatz
            The adjoint state circuit.
        state : QuantumCircuit or EvolvedOperatorAnsatz
            The state circuit.

        Returns
        -------
        float
            The overlap value.
        """
        return ovlap_hadamard_test(simulator = self.simulator, 
            adjoint_state = adjoint_state, 
            state = state
        )

    def get_constants(
        self,
        adjoint_state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
        state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
        operator: SparsePauliOp
        ) -> complex:
        """
        Perform both real and imaginary Hadamard tests to estimate complex scaling constants.

        Args:
            adjoint_state (QuantumCircuit or EvolvedOperatorAnsatz): Circuit representation of an adjoint basis state used define a matrix element.
            state (QuantumCircuit or EvolvedOperatorAnsatz): Circuit representation of a basis state used to define a matrix element. 
            operator (SparsePauliOp): Operator for which the expectation value is calculated.

        Returns:
            complex: Estimated complex expectation value of the operator.
        """
        return op_hadamard_test(
            simulator=self.simulator,
            adjoint_state=adjoint_state,
            state=state,
            operator=operator
        )

    def create_herm_op(self,
        state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
        operator: SparsePauliOp
        ) -> Operator:
        """
        Creating particle/hole state cost function operator to be optimised via VQE.

        Parameters
        ----------
        state : QuantumCircuit or EvolvedOperatorAnsatz
            The quantum state circuit.
        operator : Operator

        Returns
        -------
        SparsePauliOp
            The Hermitian operator.
        """
        den_mat=DensityMatrix(state)
        den_mat=den_mat.evolve(operator)
        den_mat=-den_mat
        return den_mat.to_operator()

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

        Returns
        -------
        dict
            Dictionary containing computed moments and related data.
        """
        gs_vqe = self.aer_vqe_hamiltonian(
            optimizer=optimizer,
            maxiter=maxiter,
            initial_point=initial_point
        )
        CircGS = gs_vqe['opt_circuit']
        E = gs_vqe['opt_energy']
        data = {}
        data['']= E 
        tensor = PolynomialTensor(data)
        E_term = FermionicOp.from_polynomial_tensor(tensor)
        moment = -self.h_elec + E_term
        moment = self.mapper.map(moment)
    
        gamma0_list = []    
        kai0_list = []
   
        kais = []
        gammas = []
        moments = []
 
        y = list(self.num_particles)
        y[0] -= +1
        num_particles = tuple(y)

        for i in range(self.num_orbitals):

            initial_state= HartreeFock(
            num_spatial_orbitals= self.num_orbitals,
            num_particles= num_particles , #nsite(1,1)
            qubit_mapper=self.mapper)

            ansatz = PUCCSD(reps=self.reps,
                num_particles= num_particles , # nelec
                num_spatial_orbitals=self.num_orbitals,
                initial_state= initial_state,
                qubit_mapper=self.mapper,
                )

            op=FermionicOp({'-_{}'.format(i):1.0},num_spin_orbitals=self.num_spin_orbitals)

            ''' Constructing the cost function '''
            
            hamiltonian = self.create_herm_op(CircGS,self.mapper.map(op)) 
            qubit_op = SparsePauliOp.from_operator(hamiltonian)

            result = self.aer_vqe_general(
                initial_point=ansatz.num_parameters*[np.random.uniform(0.0, 10.0)],
                ansatz=ansatz,
                qubit_hamiltonian=qubit_op,
                maxiter=maxiter,
                optimizer=optimizer
                )['opt_circuit']

            ''' Optimising the ansatz '''
            kai0 = result
            kai0 = kai0.decompose(reps=3)   
            kai0_list.append(kai0)
            
            gamma0_op = self.mapper.map(op)
            gamma0 = self.get_constants(kai0, CircGS,gamma0_op) 
            gamma0_list.append(gamma0)

        moment_matrix = np.zeros((self.num_orbitals,self.num_orbitals))

        kais.append(kai0_list)
        gammas.append(gamma0_list)

        oplist =[]
        off_diag_gamma_list = []
        moment_matrix = np.zeros((self.num_orbitals,self.num_orbitals))

        for i in range(self.num_orbitals):

            overlap_00 = self.get_ovlp(kai0_list[i],kai0_list[i])
            terms = gamma0_list[i].conjugate()*gamma0_list[i] * overlap_00
            moment_matrix[i,i] = terms.real

            op = FermionicOp({'+_{}'.format(i):1.0},num_spin_orbitals=self.num_spin_orbitals)
            
            oplist.append(op)   
            
            for j in range(self.num_orbitals):
                if i != j:
                    
                    off_diag_terms0 = self.get_constants(CircGS, kai0_list[j] ,self.mapper.map(op))    
                    off_diag_gamma_list.append(off_diag_terms0)

                    off_diag_overlap = self.get_ovlp(kai0_list[j],kai0_list[j])
                    off_diag_terms = off_diag_gamma_list[i].conjugate()*gamma0_list[j] * off_diag_overlap
                    moment_matrix[i,j] = off_diag_terms.real

        # If nmom is 0, return the density matrix and ground state energy
        if nmom == 0:
            output = {'density matrix': moment_matrix, 'ground state energy':E}
            return output

        moments.append(moment_matrix)

        def get_kais_and_gammas(previous_kai_list,previous_gamma_list): 

            new_kai_list = []
            new_gamma_list = []

            for i in range(self.num_orbitals):

                initial_state= HartreeFock(
                num_spatial_orbitals= self.num_orbitals,
                num_particles= num_particles, 
                qubit_mapper=self.mapper)

                ansatz = PUCCSD(reps=self.reps,
                num_particles= num_particles, 
                num_spatial_orbitals=self.num_orbitals,
                initial_state= initial_state,
                qubit_mapper=self.mapper,
                )

                hamiltonian = self.create_herm_op(previous_kai_list[i], moment)
                qubit_op = SparsePauliOp.from_operator(hamiltonian)
                result = self.aer_vqe_general(
                    initial_point=ansatz.num_parameters*[np.random.uniform(0.0, 10.0)],
                    ansatz=ansatz,
                    qubit_hamiltonian=qubit_op,
                    maxiter=maxiter,
                    optimizer=optimizer
                )['opt_circuit']

                ansatz = result
                kai = ansatz.decompose(reps=3)
                new_kai_list.append(kai)

                gamma_op = SparsePauliOp.from_operator(moment)
                gamma_new = self.get_constants(kai, previous_kai_list[i],gamma_op)
                gamma = previous_gamma_list[i] * gamma_new
                
                new_gamma_list.append(gamma)

            return new_kai_list, new_gamma_list
        
        def get_new_moments(ground_state,zeroth_gamma_list,zeroth_kai_list,current_gamma_list,current_kai_list):

            new_moment_matrix = np.zeros((self.num_orbitals,self.num_orbitals))  
            new_off_diag_gamma_list = []
  
            for i in range(self.num_orbitals):

                overlap_new = self.get_ovlp(zeroth_kai_list[i],current_kai_list[i])
                terms = zeroth_gamma_list[i].conjugate()*current_gamma_list[i] *  overlap_new
                new_moment_matrix[i,i] = terms.real    
                                         
                for j in range(self.num_orbitals):

                    if i != j:
                        op = FermionicOp({'+_{}'.format(i):1.0},num_spin_orbitals=self.num_spin_orbitals)
                        
                        off_diag_terms = self.get_constants(ground_state, current_kai_list[j] ,self.mapper.map(op))    
                        new_off_diag_gamma_list.append(off_diag_terms)
        
                        overlap = self.get_ovlp(current_kai_list[j],current_kai_list[j])
                        terms = new_off_diag_gamma_list[i].conjugate()*current_gamma_list[j] *  overlap
                        new_moment_matrix[i,j] = terms.real         

            return new_moment_matrix

        for j in range(nmom):

            kai_list, gamma_list = get_kais_and_gammas(kais[j],gammas[j])
            kais.append(kai_list)
            gammas.append(gamma_list)
            new_moments = get_new_moments(CircGS,gammas[0],kais[0],gamma_list,kai_list)
            moments.append(new_moments)
            
        moments = np.concatenate(moments, axis=0).reshape(nmom+1,self.num_orbitals,self.num_orbitals)
        outputs = {'moments': moments,'kais': kais,'gammas': gammas}

        return outputs

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
        gs_vqe = self.aer_vqe_hamiltonian(
            optimizer=optimizer,
            maxiter=maxiter,
            initial_point=initial_point
        )
        CircGS = gs_vqe['opt_circuit']
        E = gs_vqe['opt_energy']
        data = {}
        data['']= E
        tensor = PolynomialTensor(data)
        E_term = FermionicOp.from_polynomial_tensor(tensor)
        moment = self.h_elec - E_term
        moment = self.mapper.map(moment)
    
        gamma0_list = []    
        kai0_list = []
   
        kais = []
        gammas = []
        moments = []
 
        y = list(self.num_particles)
        y[0] += 1
        num_particles = tuple(y)

        for i in range(self.num_orbitals):

            initial_state= HartreeFock(
            num_spatial_orbitals= self.num_orbitals,
            num_particles= num_particles , #nsite(1,1)
            qubit_mapper=self.mapper)

            ansatz = PUCCSD(reps=self.reps,
                num_particles= num_particles , # nelec
                num_spatial_orbitals=self.num_orbitals,
                initial_state= initial_state,
                qubit_mapper=self.mapper,
                )

            op=FermionicOp({'+_{}'.format(i):1.0},num_spin_orbitals=self.num_spin_orbitals)

            ''' Constructing the cost function '''
            hamiltonian = self.create_herm_op(CircGS,self.mapper.map(op)) 
            qubit_op = SparsePauliOp.from_operator(hamiltonian)

            result = self.aer_vqe_general(
                initial_point=ansatz.num_parameters*[np.random.uniform(0.0, 10.0)],
                ansatz=ansatz,
                qubit_hamiltonian=qubit_op,
                maxiter=maxiter,
                optimizer=optimizer
            )['opt_circuit']

            kai0 = result
            kai0 = kai0.decompose(reps=3)   
            kai0_list.append(kai0)
            
            gamma0_op = self.mapper.map(op)
            gamma0 = self.get_constants(kai0, CircGS,gamma0_op) 
            gamma0_list.append(gamma0)

        moment_matrix = np.zeros((self.num_orbitals,self.num_orbitals))

        kais.append(kai0_list)
        gammas.append(gamma0_list)

        oplist =[]
        off_diag_gamma_list = []
        moment_matrix = np.zeros((self.num_orbitals,self.num_orbitals))

        for i in range(self.num_orbitals):

                overlap_00 = self.get_ovlp(kai0_list[i],kai0_list[i])
                terms = gamma0_list[i].conjugate()*gamma0_list[i] * overlap_00
                moment_matrix[i,i] = terms.real
                op = FermionicOp({'-_{}'.format(i):1.0},num_spin_orbitals=self.num_spin_orbitals)
                oplist.append(op)   
                
                for j in range(self.num_orbitals):

                    if i != j:         

                        off_diag_terms0 = self.get_constants(CircGS, kai0_list[j] ,self.mapper.map(op))    
                        off_diag_gamma_list.append(off_diag_terms0)

                        off_diag_overlap = self.get_ovlp(kai0_list[j],kai0_list[j])
                        off_diag_terms = off_diag_gamma_list[i].conjugate()*gamma0_list[j] * off_diag_overlap
                        moment_matrix[i,j] = off_diag_terms.real

        moments.append(moment_matrix)

        def get_kais_and_gammas(previous_kai_list,previous_gamma_list): 

            new_kai_list = []
            new_gamma_list = []

            for i in range(self.num_orbitals):

                initial_state= HartreeFock(
                num_spatial_orbitals= self.num_orbitals,
                num_particles= num_particles , #nsite(1,1)
                qubit_mapper=self.mapper)

                ansatz = PUCCSD(reps=self.reps,
                num_particles= num_particles , # nelec
                num_spatial_orbitals=self.num_orbitals,
                initial_state= initial_state,
                qubit_mapper=self.mapper,
                )

                hamiltonian = self.create_herm_op(previous_kai_list[i], moment)
                qubit_op = SparsePauliOp.from_operator(hamiltonian)

                result = self.aer_vqe_general(
                    initial_point=ansatz.num_parameters*[np.random.uniform(0.0, 10.0)],
                    ansatz=ansatz,
                    qubit_hamiltonian=qubit_op,
                    maxiter=maxiter,
                    optimizer=optimizer
                )['opt_circuit']

                ansatz = result
                kai = ansatz.decompose(reps=3)
                new_kai_list.append(kai)

                gamma_op = SparsePauliOp.from_operator(moment)
                gamma_new = self.get_constants(kai, previous_kai_list[i],gamma_op)
                gamma = previous_gamma_list[i] * gamma_new
                
                new_gamma_list.append(gamma)

            return new_kai_list, new_gamma_list
        
        def get_new_moments(ground_state,zeroth_gamma_list,zeroth_kai_list,current_gamma_list,current_kai_list):

            new_moment_matrix = np.zeros((self.num_orbitals,self.num_orbitals))  
            new_off_diag_gamma_list = []
  
            for i in range(self.num_orbitals):

                overlap_new = self.get_ovlp(zeroth_kai_list[i],current_kai_list[i])
                terms = zeroth_gamma_list[i].conjugate()*current_gamma_list[i] *  overlap_new
                new_moment_matrix[i,i] = terms.real    
                                         
                for j in range(self.num_orbitals):

                    if i != j:
                        op = FermionicOp({'-_{}'.format(i):1.0},num_spin_orbitals=self.num_spin_orbitals)
                        
                        off_diag_terms = self.get_constants(ground_state, current_kai_list[j] ,self.mapper.map(op))    
                        new_off_diag_gamma_list.append(off_diag_terms)
        
                        overlap = self.get_ovlp(current_kai_list[j],current_kai_list[j])
                        terms = new_off_diag_gamma_list[i].conjugate()*current_gamma_list[j] *  overlap
                        new_moment_matrix[i,j] = terms.real         

            return new_moment_matrix

        for j in range(nmom):

            kai_list, gamma_list = get_kais_and_gammas(kais[j],gammas[j])
            kais.append(kai_list)
            gammas.append(gamma_list)
            new_moments = get_new_moments(CircGS,gammas[0],kais[0],gamma_list,kai_list)
            moments.append(new_moments)
            
        moments = np.concatenate(moments, axis=0).reshape(nmom+1,self.num_orbitals,self.num_orbitals)
        outputs = {'moments': moments,'kais': kais,'gammas': gammas}

        return outputs

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
