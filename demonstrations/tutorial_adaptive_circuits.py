r"""

Adaptive circuits for quantum chemistry
=======================================

.. meta::
    :property="og:description": Learn how to build quantum chemistry circuits adaptively

    :property="og:image": https://pennylane.ai/qml/_images/adaptive_circuits.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief ofverview of VQE


*Author: PennyLane dev team. Posted:  2021. Last updated: *

Quantum computers promise a significant speedup in simulating molecular electronic structures for
systems that are intractable with classical quantum chemistry algorithms. The variational quantum
eigensolver is the method of choice for performing such quantum chemistry simulations on near-term
quantum devices. The key component of the VQE algorithm is to prepare the quantum state describing
the ground electronic state of a molecule on a quantum device.

Preparation of the state requires a quantum circuit that is composed of a certain number of gates
applied to an initial reference state. For a given molecule, the appropriate circuit can be
generated by using a pre-selected wavefunction ansatz or it can be generated via a specialized
approach that builds a customized ansatz for the given molecule. Using a pre-constructed ansatz has
the disadvantage of sacrificing performance for generality while building a customized circuit
helps with improving performance in the cost of reducing generality.

In this tutorial you will learn how to build customized quantum chemistry circuits adaptively. This
includes a recipe for adaptive selection of the gates that have a significant contribution to the
desired state and neglecting those which have zero or small contributions. You also learn to use
the functionality in PennyLane for leveraging the sparsity of a molecular Hamiltonian to make the
computation of the expectation values even more efficient. Let’s get started!

Quantum chemistry circuits
--------------------------

Quantum chemistry circuits are typically build using a pre-generated wavefunction ansatz such as
UCCSD. In this approach, one includes all possible single and double excitations of electrons from
the occupied spin-orbitals of a reference state to the unoccupied spin-orbitals. This makes
construction of the ansatz straightforward for any given molecule. However, we know that in
practical applications, only a  selected number of such excitations are necessary to prepare the
ground state wavefunction and including all possible excitations increases the cost of the
simulations without improving the accuracy of the results. This motivates implementing an strategy
that allows approximating the contribution of the excitations and select only those excitations
that are found to be important for the given molecule. This can be done by using adaptive methods
to construct a circuit for each given problem.

Adaptive circuits
-----------------

The main idea behind building adaptive circuits is to compute the gradients with respect to
excitation gates and select gates based on the magnitude of the computed gradients. There are
different ways to select the gates based on the computed gradients and here we apply two of these
strategies to compute the ground state energy of the trihydrogen cation. Both of these methods
require constructing the Hamiltonian and determine all possible excitations. We first import the
required libraries and define the molecular parameters. Note that the atomic coordinates are in Bohr.
"""

import numpy as np
import pennylane as qml
from pennylane import qchem

symbols = ["H", "H", "H"]
geometry = np.array([0.01076341,  0.04449877, 0.00000000,
                     0.98729511,  1.63059090, 0.00000000,
                     1.87262411, -0.00815842, 0.00000000])

##############################################################################
# Then we compute the the molecular Hamiltonian and the electronic excitations.

H, qubits = qchem.molecular_hamiltonian(
    symbols,
    geometry,
    charge = 1,
    mult = 1,
    basis = 'sto-3g',
    active_electrons = 2,
    active_orbitals  = 3)

singles, doubles = qchem.excitations(2, qubits)

##############################################################################
# We first implement a strategy which constructs the circuit by adding groups of gate at a time.
# The first step is to compute the gradients for a set of the existing gates. We construct our first
# group by including all of the double excitations anc create a circuit that applies those gates to
# a reference Hartree-Fock state.

def circuit_1(params, wires, excitations):
    qml.BasisState(np.array([1, 1, 0, 0, 0, 0], requires_grad=False), wires=wires)
    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(params[i], wires=excitation)
        else:
            qml.SingleExcitation(params[i], wires=excitation)

##############################################################################
# We now compute the gradients for each of the double excitation gates. We also need to define a
# device and a cost function.

dev = qml.device("default.qubit", wires = qubits)
cost_fn = qml.ExpvalCost(circuit_1, H, dev, optimize=True)

dcircuit = qml.grad(cost_fn, argnum=0)

params = [0.0] * len(doubles)
grads = dcircuit(params, excitations=doubles)

print(grads)

##############################################################################
# The computed gradients have different values that are a measure of the contribution of each gate
# in the final state prepared by the circuit. We select those gates that have a gradient above a
# pre-defined threshold which we set to 0.001.

doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > 0.001]

##############################################################################
# We add the selected gates to the circuit and perform one VQE step to determine the optimized
# parameters for the selected gates. We also need to define an optimizer. Note that the VQE
# optimization is not very costly as we only have two gate in our circuit.

opt = qml.GradientDescentOptimizer(stepsize=0.5)

params_doubles = [0.0] * len(doubles_select)

for n in range(10):
    params_doubles, energy = opt.step_and_cost(cost_fn, params_doubles, excitations=doubles_select)
    print(energy)

##############################################################################
# Now, we keep the selected gates in the circuit and compute the gradients with respect to all of
# the single excitation gates and select those that have a non-negligible gradient. To do that, we
# need to slightly modify our circuit such that parameters of the double excitation gates are kept
# fixed while the gradients are computed for the single excitation gates.

def circuit_2(params, wires, excitations, gates_select, params_select):
    qml.BasisState(np.array([1, 1, 0, 0, 0, 0], requires_grad=False), wires=wires)
    for i, gate in enumerate(gates_select):
        if len(gate) == 4:
            qml.DoubleExcitation(params_select[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params_select[i], wires=gate)

    for i, gate in enumerate(excitations):
        if len(gate) == 4:
            qml.DoubleExcitation(params[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params[i], wires=gate)

##############################################################################
#  We now compute the gradients for the single excitation gates.

cost_fn = qml.ExpvalCost(circuit_2, H, dev, optimize=True)

dcircuit = qml.grad(cost_fn, argnum=0)

params = [0.0] * len(singles)

grads = dcircuit(params, excitations=singles, gates_select=doubles_select, params_select=params_doubles)

##############################################################################
# Similar to the double excitation gates, we select those single excitations that have a gradient
# larger than a predefined threshold.

singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > 0.001]

##############################################################################
# Now, we have all the gates we need to build our circuit and perform one final step of VQE
# optimization to get the ground state energy. The resulting energy is the exact energy of the
# ground state.

cost_fn = qml.ExpvalCost(circuit_1, H, dev, optimize=True)

params = [0.0] * len(doubles_select + singles_select)

for n in range(10):
    params, energy = opt.step_and_cost(cost_fn, params, excitations=doubles_select + singles_select)
    print(energy)

##############################################################################
# Success! We can obtain the exact ground state energy of the terihydrogen cation indicates by
# having only two gates in the circuit.

##############################################################################
# Sparse method
# Molecular Hamiltonian and quantum states are sparse. For instance, let’s look at the Hamiltonian
# we built for H3+. We can compute the matrix representation of the Hamiltonian in the computational
# basis using the sparse_hamiltonian function. This function returns the matrix in the scipy sparse
# coordinate format.

H_sparse = qml.utils.sparse_hamiltonian(H)
print(H_sparse)

##############################################################################
# We can already see that the matrix has 64x64 elements which only 304 of them
# are non-zero. Leveraging this sparsity in the VQE simulations can significantly reduce the
# simulation times. We use the implemented functionality in PennyLane for computing the expectation
# value of the sparse Hamiltonian observable. This can reduce the cost of the VQE simulations by
# orders of magnitude depending on the molecular size. We use the selected gates in the previous
# steps and perform the final VQE step with the sparse method.

dev = qml.device("default.qubit", wires=qubits)
opt = qml.GradientDescentOptimizer(stepsize=0.5)
params = [0.0] * (len(doubles) + len(singles))

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params, wires, H_sparse):
    qml.BasisState(np.array([1, 1, 0, 0, 0, 0], requires_grad=False), wires=wires)

    for i, excitation in enumerate(doubles):
        qml.DoubleExcitation(params[i], wires=excitation)

    for j, excitation in enumerate(singles):
        qml.SingleExcitation(params[i + j + 1], wires=excitation)

    return qml.expval(qml.SparseHamiltonian(H_sparse, wires=wires))

def cost(params, wires, H_sparse):
    return circuit(params, wires, H_sparse)

for n in range(20):
    params, energy = opt.step_and_cost(cost, params, wires, H_sparse)
    print("n = {:},  E = {:.8f} H".format(n, energy))

##############################################################################
# Using the sparse method reproduces the exact ground state energy while the optimization time is
# reduced. The performance of the optimization will be more significant for larger molecules.
#
# In conclusion, we have leaned that building quantum chemistry circuits adaptively and using the
# functionality for sparse objects makes molecular simulations very efficient.
#
# References
# ----------
#
#