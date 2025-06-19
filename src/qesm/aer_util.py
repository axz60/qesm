### qiskit circuit construction and typing modules ####
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile
from qiskit.circuit.library import EvolvedOperatorAnsatz, UnitaryGate
from qiskit.quantum_info import Statevector,SparsePauliOp
from qiskit.quantum_info.operators import Operator


### qiskit measurement modules ###
from qiskit_algorithms.optimizers import SPSA , L_BFGS_B, SLSQP , COBYLA , POWELL
from qiskit_algorithms import VQE
from qiskit.primitives import BaseEstimatorV1, BaseEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator


######### others #########
import numpy as np
import scipy.linalg
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from typing import Any, Dict, Union

'''

This module provides utility functions to aide 
direct/recursive spectral moments algorithms using Qiskit Aer. 

'''

def run_vqe(
    estimator: BaseEstimatorV1,                      
    ansatz: Union[QuantumCircuit, EvolvedOperatorAnsatz],                        
    qubit_hamiltonian: SparsePauliOp,    
    optimizer: str = 'slsqp',            
    maxiter: int = 100,
    initial_point: Union[None, list, np.ndarray] = None,
    nuclear_repulsion_energy: float = 0.0
    ) -> Dict[str, Any]:

    """
    Run the VQE algorithm with the provided parameters.

    Args:
        estimator (BaseEstimatorV1): Qiskit-Aer based estimator for sampling cost functions.
        ansatz (QuantumCircuit or EvolvedOperatorAnsatz): Variational ansatz circuit for VQE.
        qubit_hamiltonian (SparsePauliOp): Hamiltonian in qubit representation.
        optimizer (str, optional): Optimizer to use 
                                    ('spsa', 'slsqp', 'l-bfgs-b', 'cobyla'). Default is 'slsqp' 
                                    and 'spsa' included for shot based cost function sampling.
        maxiter (int, optional): Maximum optimizer iterations. Default is 100.
        initial_point (list or np.ndarray or None, optional): Initial parameter values. Default is None.
        nuclear_repulsion_energy (float, optional): Nuclear repulsion energy to add to VQE result. Default is 0.0.

    Returns:
        Dict[str, Any]: Dictionary containing the VQE results:
            - "opt_params": optimal parameters found
            - "opt_circuit": parameterized circuit with optimal parameters
            - "opt_objective": minimum value of the objective function
            - "opt_energy": total ground state energy
    """
    if not isinstance(estimator, BaseEstimatorV1):
        raise NotImplementedError("This function currently only supports BaseEstimatorV1 from Qiskit Aer.")
        
    # Normalize optimizer string to lower case for easier matching
    optimizer = optimizer.lower()

    if optimizer in ['spsa']:
        opt = SPSA(maxiter=maxiter)
    elif optimizer in ['slsqp']:
        opt = SLSQP(maxiter=maxiter)
    elif optimizer in ['lb', 'l-bfgs-b', 'lbfgsb']:
        opt = L_BFGS_B(maxiter=maxiter)
    elif optimizer in ['cobyla', 'cob']:
        opt = COBYLA(maxiter=maxiter)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    vqe = VQE(estimator, ansatz, optimizer=opt, initial_point=initial_point)
    result = vqe.compute_minimum_eigenvalue(operator=qubit_hamiltonian)

    output = {
        "opt_params": result.optimal_point,
        "opt_circuit": ansatz.assign_parameters(result.optimal_point),
        "opt_objective": result.optimal_value,
        "opt_energy": result.optimal_value + nuclear_repulsion_energy,
    }
    return output

def aer_expectation_value(
    estimator: BaseEstimatorV1,
    circuit: Union[QuantumCircuit, EvolvedOperatorAnsatz], 
    qubit_hamiltonian: SparsePauliOp,
    ) -> float:

    """
    Calculate the expectation value of a Hamiltonian using the provided estimator.

    Args:
        estimator (BaseEstimatorV1): Qiskit-Aer based estimator for sampling expectation values.
        circuit (QuantumCircuit or EvolvedOperatorAnsatz): State used to evaluate expectation values.
        qubit_hamiltonian (SparsePauliOp): Hamiltonian in qubit representation.

    Returns:
        float: The expectation value of the Hamiltonian.
    """
    if not isinstance(estimator, BaseEstimatorV1):
        raise NotImplementedError("This function currently only supports BaseEstimatorV1 from Qiskit Aer.")
    
    job = estimator.run(circuit, qubit_hamiltonian)

    return job.result().values[0]

def hadamard_test_simulator(
    simulator: AerSimulator,
    adjoint_state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
    state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
    real: bool,
    ovlap: bool,
    operator: SparsePauliOp = None,
    ) -> float:

    """
    Performs a real/imaginary Hadamard test to estimate real/imaginary parts of expectation values/matrix elements and overlaps between two quantum states.

    Args:
        simulator (AerSimulator): Qiskit-Aer based circuit simulator (Not be confused with Qiskit/Aer Sampler primatices).
        adjoint_state (QuantumCircuit or EvolvedOperatorAnsatz): Circuit representation of an adjoint (bra) state.
        state (QuantumCircuit or EvolvedOperatorAnsatz): Circuit representation of a (ket) state. 
        operator (SparsePauliOp): Operator used to estimate matrix element. If None, then the Hadamard test is used to estimate the overlap between two states.
        real (bool): If True, then real Hadamard test is performed. If False, imaginary Hadamard test is performed.
        ovlap (bool): If True, then the Hadamard test is used to estimate the overlap between two states. If False, it estimates the expectation value/matrix element given an operator.

    Returns:
        float: Estimated real/imaginary expectation value/matrix element given an operator or the overlap between two states.
    """
    if not isinstance(real, bool):
        raise ValueError("'real' must be a boolean value (True or False), to indicate whether to perform a real or imaginary Hadamard test.")
    if not isinstance(ovlap, bool):
        raise ValueError("'ovlap' must be a boolean value (True or False), to indicate whether to estimate matrix element or overlap via real/imaginary Hadamard test.")
    if ovlap is True: 
        assert operator is None, "If 'ovlap' is True, then 'operator' must be None to estimate the overlap between two states."
    elif ovlap is False:
        assert operator is not None, "If 'ovlap' is False, then 'operator' must be provided to estimate the expectation value or matrix element given an operator."

    shots = simulator.options.shots
    num_qubits = adjoint_state.num_qubits
    quantum_register = QuantumRegister(num_qubits+1)
    classical_register = ClassicalRegister(1)
    circuit = QuantumCircuit(quantum_register,classical_register)    

    # Constructing the controled unitary gate for the Hadamard test

    def closest_unitary(A):
        """ 
        Find the closest unitary matrix to the given matrix A.
        Uses Singular Value Decomposition (SVD) to find the closest unitary.
        Returns a matrix.

        Args:
            A (np.ndarray): Input matrix to find the closest unitary for.
        Returns:
            np.ndarray: The closest unitary matrix to A.
        """
        V, __, Wh = scipy.linalg.svd(A)
        U = V @ Wh
        return U

    def make_control_gate(U):
        """ 
        Make a controlled gate from a QISKIT quantum circuit, U. Return the 
        controlled gate as a new UnitaryGate object.
        Args:
            U (QuantumCircuit): Unitary matrix to create a controlled gate from.
        Returns:
            Unitary Gate (UnitaryGate): A controlled unitary gate based on the input matrix U.
        """
        u_mat = Operator(U).to_matrix()

        I = np.eye(np.shape(u_mat)[0])
        zeros = np.zeros(np.shape(u_mat))
        controlled_unitary = np.block([[I, zeros], [zeros, u_mat]])

        gate = UnitaryGate(Operator(closest_unitary(controlled_unitary)))
        gate_circ = QuantumCircuit(gate.num_qubits)
        gate_circ.append(gate,list(range(gate.num_qubits)))

        return UnitaryGate(Operator(closest_unitary(Operator(gate_circ.reverse_bits()).to_matrix())))

    if not ovlap:
        unitary_operator = Operator(closest_unitary(operator))    
    unitary_state1 = UnitaryGate(Operator(closest_unitary(Operator(state).to_matrix())))
    unitary_state2 = UnitaryGate(Operator(closest_unitary(Operator(adjoint_state).to_matrix())))

    qc_temp = QuantumCircuit(num_qubits,name='overlap_unitary')
    qc_temp.append(unitary_state1,list(range(num_qubits)))
    if not ovlap:
        qc_temp.append(unitary_operator,list(range(num_qubits)))
    qc_temp.append(unitary_state2.adjoint(),list(range(num_qubits)))
    c_unitary = make_control_gate(qc_temp)

    # constructing the hadamard test circuit

    circuit.h(0)
    if not real:
        circuit.sdg(0)
    circuit.append(c_unitary,list(range(num_qubits+1)))
    circuit.h(0)
    circuit.measure(quantum_register[0],classical_register[0]) 

    # Transpile the circuit for the simulator backend and run it

    transpilied_circ  = transpile(circuit,simulator,optimization_level=0)
    probs = simulator.run(transpilied_circ).result().get_counts()

    if len(probs) == 1:
        try:
            probs['0']
            mean_val = 1 
        except:
            mean_val = -1
    else :
        mean_val = (probs['0'] - probs['1'])/shots

    return mean_val

def op_hadamard_test(
    simulator: Any,
    adjoint_state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
    state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
    operator: SparsePauliOp
    ) -> complex:

    """
    Perform both real and imaginary Hadamard tests to estimate complex matrix elements

    Args:
        simulator (AerBackend): Qiskit-Aer based circuit simulator (Not be confused with Qiskit any Sampler primative).
        adjoint_state (QuantumCircuit or EvolvedOperatorAnsatz): Circuit representation of an adjoint basis state used define a matrix element.
        state (QuantumCircuit or EvolvedOperatorAnsatz): Circuit representation of a basis state used to define a matrix element. 
        operator (SparsePauliOp): Operator used to estimate matrix element.
        
    Returns:
        complex: Estimated complex expectation value of the operator.

    """
    qubit_op = SparsePauliOp.from_operator(operator)
    expr = []

    for j in range(len(qubit_op.paulis)):

        expectation_real = hadamard_test_simulator(simulator,adjoint_state,state,operator = qubit_op.paulis[j],real=True,
                                                      ovlap=False)
        expectation_imag = hadamard_test_simulator(simulator,adjoint_state,state, operator = qubit_op.paulis[j],real=False,
                                                      ovlap=False)
        mean_val = qubit_op.coeffs[j] * expectation_real + 1j*qubit_op.coeffs[j] * expectation_imag
        expr.append(mean_val)
              
    return sum(expr)

def ovlap_hadamard_test(
    simulator: Any,
    adjoint_state: Union[QuantumCircuit, EvolvedOperatorAnsatz],
    state: Union[QuantumCircuit, EvolvedOperatorAnsatz]
    ) -> complex:

    """
    Perform Hadamard test to estimate the overlap between two states.

    Args:
        simulator (AerBackend): Qiskit-Aer based circuit simulator (Not be confused with Qiskit any Sampler primative).
        adjoint_state (QuantumCircuit or EvolvedOperatorAnsatz): Circuit representation of an adjoint basis state used define a matrix element.
        state (QuantumCircuit or EvolvedOperatorAnsatz): Circuit representation of a basis state used to define a matrix element. 

    Returns:
        float: Estimated overlap between the two states.
    """
    real = hadamard_test_simulator(
        simulator, adjoint_state, state, operator=None, real=True, ovlap=True
    )
    imag = hadamard_test_simulator(
        simulator, adjoint_state, state, operator=None, real=False, ovlap=True
    )
    return real + 1j * imag

def primatives(shots: int = None) -> Dict[str, Any]:
    """
    Create and return Qiskit Aer primitives for VQE and expectation value calculations.
    Args:
        shots (int, optional): Number of shots for sampling. If None, uses statevector simulation. Default is None.
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'estimator': Qiskit Aer EstimatorV1 for sampling cost functions.
            - 'simulator': Qiskit Aer simulator for statevector simulation.
            - 'sv estimator': Qiskit Aer EstimatorV1 for statevector simulation.
    
    Notes:
        This is mainly for convenience/testing; consider using Qiskit Aer primitives
        directly for production code.
    """
    qasm_estimator = AerEstimator(
        run_options={"shots": shots}, approximation=True
    )
    sv_estimator = AerEstimator(
        run_options={"shots": None}, approximation=True
    )
    simulator = AerSimulator(method='statevector', shots=shots)

    output = {
        'estimator': qasm_estimator,
        'simulator': simulator,
        'sv estimator': sv_estimator,
    }
    return output