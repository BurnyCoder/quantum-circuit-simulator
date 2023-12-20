import numpy as np

# Constant for 1/sqrt(2), commonly used in quantum computing, especially in the Hadamard gate.
isq2 = 1.0 / (2.0**0.5)

class Qstate:
    def __init__(self, n):
        # n: Number of qubits. A qubit is the basic unit of quantum information, analogous to a bit in classical computing.
        # In this code, a qubit is represented as part of a state vector of size 2^n.
        # For example, in a 2-qubit system, the state vector has 4 elements, each representing a quantum state like |00>, |01>, |10>, |11>.
        self.n = n
        self.state = np.zeros(2**self.n, dtype=complex)  # State vector initialized in the |0...0> state.
        self.state[0] = 1

    def op(self, t, i):
        # General method to apply a quantum gate to the i-th qubit.
        # Quantum gates are represented by matrices, and their action on a qubit is represented by matrix multiplication.
	# In a multi-qubit system, to apply a gate to a specific qubit, two identity matrices are used: eyeL and eyeR. 
	# eyeL is an identity matrix that corresponds to the tensor product space before the target qubit.
	# eyeR is an identity matrix for the tensor product space after the target qubit.
	# These identity matrices, when used in a Kronecker product with the gate matrix, ensure that the gate only affects the target qubit while leaving the states of all other qubits unchanged.
        eyeL = np.eye(2**i, dtype=complex)  # Identity matrix for left side of tensor product.
        eyeR = np.eye(2**(self.n - i - int(t.shape[0]**0.5)), dtype=complex)  # Identity matrix for right side.
        t_all = np.kron(np.kron(eyeL, t), eyeR)  # Full operation matrix using tensor product, Kronecker product of matrices A and B, denoted A⊗B, is a block matrix where each element of A is multiplied by the whole matrix B.
        self.state = np.matmul(t_all, self.state)  # Apply the transformation.

    def hadamard(self, i):
        # Hadamard gate: Creates superpositions. It maps |0> to (|0> + |1>)/sqrt(2) and |1> to (|0> - |1>)/sqrt(2).
        h_matrix = isq2 * np.array([
            [1, 1],
            [1, -1]
        ], dtype=complex)
        self.op(h_matrix, i)  # Apply to i-th qubit.

    def t(self, i):
        # T gate: Adds a phase of π/4. It's a phase shift gate that changes the phase of the |1> state.
        t_matrix = np.array([
            [1, 0],
            [0, isq2 + isq2 * 1j]
        ], dtype=complex)
        self.op(t_matrix, i)  # Apply to i-th qubit.

    def s(self, i):
        # S gate: Similar to the T gate, but adds a phase of π/2.
        s_matrix = np.array([
            [1, 0],
            [0, 0 + 1j]
        ], dtype=complex)
        self.op(s_matrix, i)  # Apply to i-th qubit.

    def cnot(self, i):
        # CNOT gate (Controlled-NOT): A two-qubit gate that flips the second (target) qubit if the first (control) qubit is |1>.
        # It is essential for creating entanglement between qubits.
        cnot_matrix = np.array([
            [1, 0, 0, 0],  # If control qubit is |0>, do nothing.
            [0, 1, 0, 0],
            [0, 0, 0, 1],  # If control qubit is |1>, flip target qubit.
            [0, 0, 1, 0]
        ], dtype=complex)
        self.op(cnot_matrix, i)  # Apply to i-th qubit as control.

    def swap(self, i):
        # SWAP gate: Swaps the states of two qubits. 
        # It changes the state |01> to |10> and vice versa, while leaving |00> and |11> unchanged.
        swap_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        self.op(swap_matrix, i)  # Apply to i-th and (i+1)-th qubits.

def measurement_function(qstate):
    # Calculate probabilities of each basis state in the state vector.
    probabilities = np.abs(qstate.state) ** 2
    # Randomly select a state based on the calculated probabilities, simulating a quantum measurement.
    return np.random.choice(len(qstate.state), p=probabilities)

def print_state(qstate):
    # Print the state vector of the quantum state.
    print("State vector:")
    for amplitude in qstate.state:
        print(amplitude)

# Test the simulator

# Initialize a 2-qubit quantum state.
qstate = Qstate(2)

# Print the initial state.
print("Initial State:")
print_state(qstate)

# Apply Hadamard gate to the first qubit and print the state.
qstate.hadamard(0)
print("\nAfter applying Hadamard to Qubit 0:")
print_state(qstate)

# Apply CNOT gate and print the state.
qstate.cnot(0)
print("\nAfter applying CNOT (Qubit 0 as control, Qubit 1 as target):")
print_state(qstate)

# Apply SWAP gate and print the state.
qstate.swap(0)
print("\nAfter swapping Qubit 0 and Qubit 1:")
print_state(qstate)

# Apply T gate to the second qubit and print the state.
qstate.t(1)
print("\nAfter applying T gate to Qubit 1:")
print_state(qstate)

# Apply S gate to the first qubit and print the state.
qstate.s(0)
print("\nAfter applying S gate to Qubit 0:")
print_state(qstate)

# Perform a measurement and print the result, including the binary representation of the measured state.
measured_state = measurement_function(qstate)
binary_format = '{0:0{1}b}'.format(measured_state, qstate.n)
print("\nMeasured state (index in state vector):", measured_state)
print(f"This means the system collapsed to the state |{binary_format}⟩ upon measurement.")
