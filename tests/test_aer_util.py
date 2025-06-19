# General libraries
import numpy as np
from qesm.aer_util import *
import pytest
import warnings
warnings.filterwarnings("ignore")

# qiskit objects
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

# Qiskit-Aer measurement objects
from qiskit_aer.primitives import Estimator as AerEstimatorV1
from qiskit_aer import AerSimulator

################################## VQE Test ##################################

# List of optimizers to test
OPTIMIZERS = ["spsa", "slsqp", "l-bfgs-b", "cobyla"]

@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_run_vqe_aer_V1(optimizer):
    # Minimal 1-qubit Hamiltonian: H = Z
    hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])

    # Simple ansatz: One-parameter Ry rotation
    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    # Aer Estimator primitive (samples on AerSimulator backend)
    
    if optimizer == "spsa":
        estimator = AerEstimatorV1(run_options={ "shots": 100000},approximation=True)
    else:
        estimator = AerEstimatorV1(run_options={ "shots": None},approximation=True)

    # VQE options
    maxiter = 100
    initial_point = [0.1]

    # Run VQE
    result = run_vqe(
        estimator=estimator,
        ansatz=qc,
        qubit_hamiltonian=hamiltonian,
        optimizer=optimizer,
        maxiter=maxiter,
        initial_point=initial_point,
        nuclear_repulsion_energy=0.0,
    )

    # Check result dictionary
    assert "opt_params" in result
    assert "opt_circuit" in result
    assert "opt_objective" in result
    assert "opt_energy" in result

    # The value will be close to -1, but allow a bit more tolerance due to shot noise
    assert isinstance(result["opt_energy"], float)
    assert result["opt_energy"] < -0.85  # Expecting close to -1, but allow some tolerance


################################## VQE Test ##################################

############################### Exp. value test ##############################

def test_expval_with_aer_estimator():
    # Pauli Z Hamiltonian: H = Z
    hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])

    # Ansatz: 1-qubit, Ry rotation, parameterized
    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    # Bind theta to 0, which yields |0> (expectation value <0|Z|0> = +1)
    qc_bound = qc.assign_parameters({theta: 0.0})

    estimator = AerEstimatorV1(run_options={ "shots": 100000},approximation=True)

    # Run your expval function
    value = aer_expectation_value(estimator, qc_bound, hamiltonian)

    # The expectation value of Z on |0> should be +1 (allow for a little noise)
    assert np.isclose(value, 1.0, atol=0.05)

    # Try theta = pi, which yields |1> (expectation value <1|Z|1> = -1)
    qc_bound = qc.assign_parameters({theta: np.pi})
    value = aer_expectation_value(estimator, qc_bound, hamiltonian)
    assert np.isclose(value, -1.0, atol=0.05)

    # Try theta = pi/2, which yields (|0> + |1>)/sqrt(2) (expectation value = 0)
    qc_bound = qc.assign_parameters({theta: np.pi/2})
    value = aer_expectation_value(estimator, qc_bound, hamiltonian)
    assert np.isclose(value, 0.0, atol=0.05)

############################### Exp. value test ##############################

#################### Operator Real/Imaginary Hadamard Test ###################

@pytest.mark.parametrize("theta,expected", [
    (0.0, 1.0),            # |0> → <0|Z|0> = +1
    (np.pi, -1.0),         # |1> → <1|Z|1> = -1
    (np.pi/2, 0.0),        # (|0>+|1>)/√2 → <+|Z|+> = 0
])
def test_op_hadamard_test_real(theta, expected):

    simulator = AerSimulator(method = 'statevector',shots = 100000)

    # Create state |psi> = Ry(theta)|0>
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    # For adjoint state, just use same state (overlap/expectation)
    adjoint_qc = qc

    # Pauli Z operator
    op = SparsePauliOp.from_list([("Z", 1.0)])

    value = hadamard_test_simulator(
        simulator=simulator,
        adjoint_state=adjoint_qc,
        state=qc,
        operator=op,
        real=True,
        ovlap=False,
    )

    assert np.isclose(value, expected, atol=0.05)

def test_op_hadamard_test_imag_zero():
    simulator = AerSimulator(method = 'statevector',shots = 100000)

    # |psi> = |0>
    qc = QuantumCircuit(1)
    qc.ry(0.0, 0)
    adjoint_qc = qc

    op = SparsePauliOp.from_list([("Z", 1.0)])

    # For Pauli Z and real state, the imaginary part is always 0
    value = hadamard_test_simulator(
        simulator=simulator,
        adjoint_state=adjoint_qc,
        state=qc,
        operator=op,
        real=False,
        ovlap=False,
    )

    assert np.isclose(value, 0.0, atol=0.05)

def test_op_hadamard_test_z_real_and_imag():
    
    simulator = AerSimulator(method = 'statevector',shots = 100000)


    # |psi> = |0>
    qc_0 = QuantumCircuit(1)
    qc_0.ry(0.0, 0)
    # |psi> = |1>
    qc_1 = QuantumCircuit(1)
    qc_1.ry(np.pi, 0)
    # |psi> = (|0> + |1>)/sqrt(2)
    qc_plus = QuantumCircuit(1)
    qc_plus.ry(np.pi/2, 0)

    op = SparsePauliOp.from_list([("Z", 1.0)])

    # Test |0>
    res = op_hadamard_test(simulator, qc_0, qc_0, op)
    assert np.isclose(np.real(res), 1.0, atol=0.05)
    assert np.isclose(np.imag(res), 0.0, atol=0.05)

    # Test |1>
    res = op_hadamard_test(simulator, qc_1, qc_1, op)
    assert np.isclose(np.real(res), -1.0, atol=0.05)
    assert np.isclose(np.imag(res), 0.0, atol=0.05)

    # Test |+>
    res = op_hadamard_test(simulator, qc_plus, qc_plus, op,)
    assert np.isclose(np.real(res), 0.0, atol=0.05)
    assert np.isclose(np.imag(res), 0.0, atol=0.05)

def test_op_hadamard_test_off_diagonal():
   
    """Test off-diagonal element <0|Z|1> (should be 0)."""
    simulator = AerSimulator(method = 'statevector',shots = 100000)


    qc_0 = QuantumCircuit(1)
    qc_0.ry(0.0, 0)
    qc_1 = QuantumCircuit(1)
    qc_1.ry(np.pi, 0)

    op = SparsePauliOp.from_list([("Z", 1.0)])

    # Off-diagonal: <0|Z|1>
    res = op_hadamard_test(simulator, qc_0, qc_1, op)
    assert np.isclose(np.real(res), 0.0, atol=0.05)
    assert np.isclose(np.imag(res), 0.0, atol=0.05)

#################### Operator Real/Imaginary Hadamard Test ###################

##################### Overlap Real/Imaginary Hadamard Test ###################
def test_ovlap_hadamard_test():

    simulator = AerSimulator(method = 'statevector',shots = 100000)

    # <0|0> = 1
    qc0 = QuantumCircuit(1)
    qc0.ry(0.0, 0)
    result = ovlap_hadamard_test(simulator, qc0, qc0)
    assert np.isclose(np.real(result), 1.0, atol=0.05)
    assert np.isclose(np.imag(result), 0.0, atol=0.05)

    # <0|1> = 0
    qc1 = QuantumCircuit(1)
    qc1.ry(np.pi, 0)
    result = ovlap_hadamard_test(simulator, qc0, qc1)
    assert np.isclose(np.real(result), 0.0, atol=0.05)
    assert np.isclose(np.imag(result), 0.0, atol=0.05)

    # <+|+> = 1 and <0|+> = 1/sqrt(2)
    qc_plus = QuantumCircuit(1)
    qc_plus.ry(np.pi/2, 0)
    result = ovlap_hadamard_test(simulator, qc_plus, qc_plus)
    assert np.isclose(np.real(result), 1.0, atol=0.05)
    result = ovlap_hadamard_test(simulator, qc0, qc_plus)
    assert np.isclose(np.real(result), 1/np.sqrt(2), atol=0.05)

##################### Overlap Real/Imaginary Hadamard Test ###################
