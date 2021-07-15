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

The key component of variational quantum algorithms for quantum chemistry is the circuit used to
prepare a quantum state describing the ground electronic state of a molecule. The variational
quantum eigensolver (VQE) [#peruzzo2014]_, [#yudong2019]_ is the method of choice for performing
such quantum chemistry simulations on near-term quantum devices. For a given molecule, the
appropriate circuit can be generated by using a pre-selected wavefunction ansatz, for example the
unitary coupled cluster ansatz with single and double excitations (UCCSD) [#romero2017]_.
Using a pre-constructed nsatz has the disadvantage of reducing performance in favour of generality:
the approach may work well in many cases, but it will not be optimized for a specific problem.
Quantum circuits can also be designed using specialized approachs that build a customized ansatz for
the given molecule. This approach helps improve performance at the cost of reducing generality.

In this tutorial, you will learn how to  **adaptively** build customized quantum chemistry circuits.
This includes a recipe for adaptive selection of the gates that have a significant contribution to
the desired state and neglecting those that have a small contribution. You also learn to use
the functionality in PennyLane for leveraging the sparsity of a molecular Hamiltonian to make the
computation of the expectation values even more efficient. Let’s get started!

Quantum chemistry circuits
--------------------------

Quantum circuits for quantum chemistry are typically build using a pre-generated wavefunction ansatz
such as UCCSD [#romero2017]_. In this approach, one includes all possible single and double
excitations of electrons from the occupied spin-orbitals of a reference state to the unoccupied
spin-orbitals [#givenstutorial]_. This makes construction of the ansatz
straightforward for any given molecule. However, in practical applications, only a selected number
of such excitations are necessary to prepare the exact ground state wavefunction. Including all
possible excitations increases the cost of the simulations without improving the accuracy of the
results. This motivates implementing a strategy that allows approximating the contribution of the
excitations and selecting only those excitations that are found to be important for the given
molecule. This can be done by using adaptive methods to construct a circuit for each given
problem [#grimsley2019]_.

Adaptive circuits
-----------------

The main idea behind building adaptive circuits is to compute the gradients with respect to all
possible excitation gates and select gates based on the magnitude of the computed gradients. 
There are different ways to select the gates based on the computed gradients. Here we discuss one of
these strategies to compute the ground state energy of LiH. This method require constructing the
Hamiltonian and determining all possible excitations, which we can do with PennyLane functionalities
shown below. But we first need to import the required libraries and define the molecular parameters
including atomic symbols and coordinates. Note that the atomic coordinates are in Bohr.
"""

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import time

symbols = ["Li", "H"]
geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])

##############################################################################
# We now compute the molecular Hamiltonian in the STO-3G basis and the electronic excitations. Here
# we restrict ourself to single and double excitations but higher-level excitations, such as
# triple and quadruple, can be considered as well. Each of these electronic excitations is
# represented by a gate that excites electrons from the occupied orbitals of a reference state to
# the unoccupied ones. This allows us to prepare a state that is a superposition of the reference
# state and all of the excited states similar to coupled cluster and configuration interaction
# methods in classical quantum chemistry.

H, qubits = qchem.molecular_hamiltonian(symbols, geometry, active_electrons=2, active_orbitals=5)

singles, doubles = qchem.excitations(2, qubits)

print(f"Total number of excitations = {len(singles) + len(doubles)}")

##############################################################################
# Note that we have a total number of 24 excitations which can be represented by the same number of
# excitation gates [#givenstutorial]_. We now implement a strategy which constructs
# the circuit by adding groups of gate at a time. We follow theses steps:
#
# 1. Compute gradients for all double excitations.
# 2. Select the double excitations with gradients larger than a pre-defined threshold.
# 3. Perform VQE to obtain the optimized parameters for the selected double excitations.
# 4. Repeats steps 1 and 2 for the single excitations.
# 5. Perform the final VQE optimization with all the selected excitations.
#
# We create a circuit that applies a selected group of gates to a reference Hartree-Fock state.


def circuit_1(params, wires, excitations):
    hf_state = qchem.hf_state(2, qubits)
    qml.BasisState(np.array(hf_state), wires=wires)
    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(params[i], wires=excitation)
        else:
            qml.SingleExcitation(params[i], wires=excitation)


##############################################################################
# We construct our first group of gates by including all of the double excitations and compute the
# gradients for each of the double excitation gates. We also need to define a device and a cost
# function. We also initialize the parameter values to zero such that the gradients are computed
# with respect to the Hartree-Fock state.


dev = qml.device("default.qubit", wires=qubits)
cost_fn = qml.ExpvalCost(circuit_1, H, dev, optimize=True)

circuit_gradient = qml.grad(cost_fn, argnum=0)

params = [0.0] * len(doubles)
grads = circuit_gradient(params, excitations=doubles)

for i in range(len(doubles)):
    print(f"Excitation : {doubles[i]}, Gradient: {grads[i]}")

##############################################################################
# The computed gradients have different values which reflect the contribution of each gate
# in the final state prepared by the circuit. Many of the gradient values are zero and we select
# those gates that have a gradient above a pre-defined threshold, which we set to 0.00001.

doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]
doubles_select

##############################################################################
# There are only 6 gates, among the total number of 16, that have gradients above the threshold.
# We add the selected gates to the circuit and perform one optimization step to determine the
# updated parameters for the selected gates. We also need to define an optimizer. Note that the
# optimization is not very costly as we only have two gate in our circuit.

opt = qml.GradientDescentOptimizer(stepsize=0.5)

params_doubles = [0.0] * len(doubles_select)

for n in range(20):
    params_doubles = opt.step(cost_fn, params_doubles, excitations=doubles_select)

##############################################################################
# Now, we keep the selected gates in the circuit and compute the gradients with respect to all of
# the single excitation gates, selecting those that have a non-negligible gradient. To do that, we
# need to slightly modify our circuit such that parameters of the double excitation gates are kept
# fixed while the gradients are computed for the single excitation gates.


def circuit_2(params, wires, excitations, gates_select, params_select):
    hf_state = qchem.hf_state(2, qubits)
    qml.BasisState(np.array(hf_state, requires_grad=False), wires=wires)
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
circuit_gradient = qml.grad(cost_fn, argnum=0)
params = [0.0] * len(singles)
grads = circuit_gradient(
    params, excitations=singles, gates_select=doubles_select, params_select=params_doubles
)
for i in range(len(singles)):
    print(f"Excitation : {singles[i]}, Gradient: {grads[i]}")

##############################################################################
# Similar to the double excitation gates, we select those single excitations that have a gradient
# larger than a predefined threshold.

singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > 1.0e-5]
singles_select

##############################################################################
# We have all of the gates we need to build our circuit and perform one final step of

# optimization to get the ground state energy. The resulting energy must be the exact energy of the
# ground electronic state of LiH which is -7.8825378193 Ha.

cost_fn = qml.ExpvalCost(circuit_1, H, dev, optimize=True)

params = [0.0] * len(doubles_select + singles_select)

for n in range(20):
    t1 = time.time()
    params, energy = opt.step_and_cost(cost_fn, params, excitations=doubles_select + singles_select)
    t2 = time.time()
    print("n = {:},  E = {:.8f} H, t = {:.2f} s".format(n, energy, t2 - t1))

##############################################################################
# Success! We obtained the exact ground state energy of LiH, within chemical accuracy, by having
# only 10 gates in our circuit. This is less than half of the total number of single and double
# excitations of LiH (24).

##############################################################################
# Sparse method
# -------------
#
# Molecular Hamiltonians and quantum states are sparse. For instance, let’s look at the Hamiltonian
# we built for LiH. We can compute its matrix representation in the computational basis using the
# function :func:`~.pennylane.utils.sparse_hamiltonian` of PennyLane. This function
# returns the matrix in the scipy 'sparse coordinate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html>'_ format.

H_sparse = qml.utils.sparse_hamiltonian(H)
H_sparse

##############################################################################
# The matrix has :math:`1024^2=1048576` entries, but only 11264 of them
# are non-zero. Leveraging this sparsity in the simulations can significantly reduce the
# simulation times. We use the implemented functionality in PennyLane for computing the expectation
# value of the sparse Hamiltonian observable. This can reduce the cost of simulations by
# orders of magnitude depending on the molecular size. We use the selected gates in the previous
# steps and perform the final optimization step with the sparse method.

dev = qml.device("default.qubit", wires=qubits)
opt = qml.GradientDescentOptimizer(stepsize=0.5)

excitations = doubles_select + singles_select
params = [0.0] * len(excitations)


@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    hf_state = qchem.hf_state(2, qubits)
    qml.BasisState(np.array(hf_state, requires_grad=False), wires=range(qubits))

    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(params[i], wires=excitation)
        elif len(excitation) == 2:
            qml.SingleExcitation(params[i], wires=excitation)

    return qml.expval(qml.SparseHamiltonian(H_sparse, wires=range(qubits)))


def cost(params):
    return circuit(params)


for n in range(20):
    t1 = time.time()
    params, energy = opt.step_and_cost(cost, params)
    t2 = time.time()
    print("n = {:},  E = {:.8f} H, t = {:.2f} s".format(n, energy, t2 - t1))

##############################################################################
# Using the sparse method reproduces the exact ground state energy while the optimization time is
# much shorter. The performance of the optimization will be more significant for larger molecules.
#
# Conclusions
# -----------
# In conclusion, we have leaned that building quantum chemistry circuits adaptively and using the
# functionality for sparse objects makes molecular simulations significantly efficient. In this
# tutorial, we followed an adaptive strategy that selects a group of gates at each time. This method
# can be extended such that the gates are selected one at time [#grimsley2019]_, or
# even to other more elaborate strategies.
#
# References
# ----------
#
# .. [#peruzzo2014]
#
#     Alberto Peruzzo, Jarrod McClean *et al.*, "A variational eigenvalue solver on a photonic
#     quantum processor". `Nature Communications 5, 4213 (2014).
#     <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# .. [#yudong2019]
#
#     Yudong Cao, Jonathan Romero, *et al.*, "Quantum Chemistry in the Age of Quantum Computing".
#     `Chem. Rev. 2019, 119, 19, 10856-10915.
#     <https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803>`__
#
# .. [#grimsley2019]
#
#     Harper R. Grimsley, Sophia E. Economou, Edwin Barnes,  Nicholas J. Mayhall, "An adaptive
#     variational algorithm for exact molecular simulations on a quantum computer".
#     `Nat. Commun. 2019, 10, 3007.
#     <https://www.nature.com/articles/s41467-019-10988-2>`__
#
# .. [#romero2017]
#
#     J. Romero, R. Babbush, *et al.*, "Strategies for quantum computing molecular
#     energies using the unitary coupled cluster ansatz". `arXiv:1701.02691
#     <https://arxiv.org/abs/1701.02691>`_
#
# .. [#givenstutorial]
#
#     :doc:`tutorial_givens_rotations`
