from data import new_problem_set
from src.utils import save_as_pickle
from src.cost_hamiltonians import max_cut_hamiltonian, max_clique_hamiltonian, max_independed_set_hamiltonian, min_vertex_cover_hamiltonian
from src import DATA_PATH

num_graphs = 50
for vertices in [4, 8, 12, 14, 16]:
    graphs = new_problem_set(num_graphs, vertices, seed=1000, pbar=True, hamiltonian=max_cut_hamiltonian)
    save_as_pickle(graphs, DATA_PATH/f"erdos_reny_v{vertices}x{num_graphs}_maxcut")

    graphs = new_problem_set(num_graphs, vertices, seed=2000, pbar=True, hamiltonian=max_independed_set_hamiltonian)
    save_as_pickle(graphs, DATA_PATH/f"erdos_reny_v{vertices}x{num_graphs}_maxindependentset")

    graphs = new_problem_set(num_graphs, vertices, seed=3000, pbar=True, hamiltonian=min_vertex_cover_hamiltonian)
    save_as_pickle(graphs, DATA_PATH/f"erdos_reny_v{vertices}x{num_graphs}_minvertexcover")

    graphs = new_problem_set(num_graphs, vertices, seed=4000, pbar=True, hamiltonian=max_clique_hamiltonian)
    save_as_pickle(graphs, DATA_PATH/f"erdos_reny_v{vertices}x{num_graphs}_maxclique")