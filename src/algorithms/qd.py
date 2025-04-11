import sys
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
from typing import List

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from src.circuit import Optimization_Circuit, Gate, I
from src.utils import nx_graph_from, to_z_basis, actions_to_gates


def prepare_circuit(circuit, solution, gate_set):
    repaired_solution = np.clip(solution, 0, len(gate_set)-0.01)
    circuit.reset()
    circuit.set_matrix_circuit(repaired_solution.reshape(circuit.width, -1))
    assert circuit.get_matrix_circuit().shape == (circuit.width,4)
    for candidate_layer in circuit.get_all_matrix_layers():
        new_layer, _ , _ = actions_to_gates(candidate_layer, gate_set)
        layer_id = len(circuit)
        sorted_gates = [new_layer.get(wire, I) for wire in range(circuit.width)]
        circuit.insert_gate_layer(layer_id, sorted_gates)
    return circuit

def qd_batch(circuit, solution_batch, gate_set, nx_graph, objective):
    objective_batch = []
    measures_batch = []
    
    for solution in solution_batch:    
        circuit = prepare_circuit(circuit, solution, gate_set)
        objective_batch.append(float(objective(to_z_basis(circuit.state()), nx_graph.edges)))
        measures_batch.append([
            len(solution[np.abs(solution) < 1]),
            sum([len(np.unique(layer[layer!=0])) for layer in circuit.get_matrix_circuit().round().astype(int).T])
        ])

    return np.array(objective_batch), np.array(measures_batch)

def qd(
        graph,
        n_steps:int, 
        objective,
        gate_set:List[Gate],
        lr:float=1.0, 
        batch_size:int=5,
        num_emitters:int=20,
        s0:float=0.5,
        show_stats:bool=True,
        show_pbar:bool=True
    ):
        problem_dim = graph.get("n_nodes")
        problem_solution = graph.get("solution").get("best_energy")
        edge_list = graph.get("graph")

        # setup
        circuit = Optimization_Circuit(problem_dim, shots=1)
        circuit_dims = problem_dim * 4 # (qubits x num_layers)
        sparcity_range = (0, circuit_dims) # (all gate positions filled, no gates set)
        uniformity_range = (5, (len(gate_set)-1)*5)
        
        # pyribs optimization archive for annealed exploring
        optimization_archive = GridArchive(solution_dim=circuit_dims,
                            dims=(sparcity_range[1], uniformity_range[1]),
                            ranges=[sparcity_range, uniformity_range],
                            learning_rate=lr,
                            threshold_min=0.0)
        
        # pyribs result archive for saving the elites
        result_archive = GridArchive(solution_dim=circuit_dims,
                            dims=(sparcity_range[1], uniformity_range[1]),
                            ranges=[sparcity_range, uniformity_range]
                            )
        
        # 'population' of emitters each with their own maintained distribution
        emitters = [
            EvolutionStrategyEmitter(
                optimization_archive,
                x0=np.ones(circuit_dims),
                sigma0=s0,
                ranker="imp",
                selection_rule="mu",
                restart_rule="basic",
                batch_size=batch_size,
            ) for _ in range(num_emitters)
        ]
        # the update algorithm, essentially
        scheduler = Scheduler(optimization_archive, emitters, result_archive=result_archive)

        # reconstruct graph
        nx_graph = nx_graph_from(problem_dim, edge_list) 

        # experiment loop
        data = []
        solution_batches = []
        objective_batches = []
        measure_batches = []
        best_solution = 0
        pbar = trange(n_steps+1, file=sys.stdout, desc='Iterations') if show_pbar else range(n_steps+1)
        for itr in pbar:
            solution_batch = scheduler.ask()
            objective_batch, measure_batch = qd_batch(circuit, solution_batch, gate_set, nx_graph, objective)
            
            solution_batches.append(solution_batch)
            objective_batches.append(objective_batch)
            measure_batches.append(measure_batch)
            
            scheduler.tell(objective_batch, measure_batch)
            if max(objective_batch) > best_solution:
                best_solution = max(objective_batch)
                pbar.set_description(f"[{itr}] AR:{best_solution/problem_solution:.1f}({best_solution:.0f}/{problem_solution:.0f})")

            data.append([itr, best_solution, problem_solution, best_solution/problem_solution, round(result_archive.stats.coverage * 100 ,3), round(result_archive.stats.norm_qd_score)])

            # Output progress every 500 iterations or on the final iteration.
            if show_stats and (itr % 50 == 0 or itr == n_steps):
                tqdm.write(f"Iteration {itr:5d} | "
                        f"Archive Coverage: {result_archive.stats.coverage * 100:6.3f}%  "
                        f"Normalized QD Score: {result_archive.stats.norm_qd_score:6.3f}")

        df = pd.DataFrame(data, columns=["step_id", "best_fit", "best_solution", "approximation_ratio", "archive_coverage", "QD_score"])
        return df, optimization_archive, result_archive#, np.array(solution_batches), np.array(objective_batches), np.array(measure_batches)