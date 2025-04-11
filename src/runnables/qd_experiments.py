import pandas as pd
import time
from data import ladder, barbell, caveman, random
from src import TEST_PATH, DATA_PATH
from src.algorithms.qd import qd
from src.cost_hamiltonians import max_clique_hamiltonian, max_cut_hamiltonian, max_independed_set_hamiltonian, min_vertex_cover_hamiltonian
from src.utils import cd, seed_all, load_pickle, save_as_pickle
from src.utils import TINY, TINYH, CLIFFORDT, ROTHCNOT, IRX

if True:
    n_steps = 100
    for vertices in [8, 12, 14, 16]:
        for shorthand, dataset_name, hamiltonian_cost_objective in [
            ('maxCUT_test', 'maxcut', max_cut_hamiltonian),
            ('maxIND_test', 'maxindependentset', max_independed_set_hamiltonian),
            ('minVER_test', 'minvertexcover', min_vertex_cover_hamiltonian),
            ('maxCLI_test', 'maxclique', max_clique_hamiltonian),
        ]:
            for cma_version, lr in [
                (f"cma-mae", 0.3),
                (f"cma-ma", 1.0),
                (f"cma-es", 0.0),
            ]:
                exp_name = f"qd_{cma_version}_er_{vertices}x50_{shorthand}"
                graphs = load_pickle(f"{DATA_PATH}/erdos_reny_v{vertices}x50_{dataset_name}")
                
                with cd(TEST_PATH / exp_name, assert_exists=False, remove=True):
                    dfs = []
                    start_time = time.time()
                    for graph_id, graph in enumerate(graphs): 
                        seed_all(graph_id)
                        df, o_archive, r_archive = qd(
                            graph, 
                            n_steps, 
                            hamiltonian_cost_objective, 
                            TINY,
                            lr=lr,
                            batch_size=2,
                            show_stats=False
                        )
                        df["name"] = f"{cma_version}"
                        df["problem"] = shorthand
                        df["graph"] = graph.get("name")
                        df["seed"] = graph_id
                        df.to_csv(f"qd_{vertices}x50_{graph_id}.csv", index=False)
                        dfs.append(df)
                        save_as_pickle({
                            "objective_archive": o_archive,
                            "result_archive": r_archive
                        }, f"{exp_name}_archives_{graph_id}")

                    print(f"took {(time.time() - start_time) / 60:.0f} minutes total")
                    save_path = f"{exp_name}_all.csv"
                    pd.concat(dfs, ignore_index=True).to_csv(save_path, index=False)
                    print(f"saved df to {save_path}")
