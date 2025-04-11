from src.utils import nx_graph_from

def max_cut_hamiltonian(assignment, graph_edges):
    return -(sum([0.5 * (int(assignment[z_i]) * int(assignment[z_j]) - 1) for z_i, z_j in graph_edges]))

def min_vertex_cover_hamiltonian(assignment, graph_edges):
    edge_penalty = sum([int(assignment[z_i]) * int(assignment[z_j]) + int(assignment[z_i]) + int(assignment[z_j]) for z_i, z_j in graph_edges])
    vertex_penality = sum([int(i) for i in assignment] )
    return -(3*edge_penalty - vertex_penality)

def max_independed_set_hamiltonian(assignment, graph_edges):
    edge_penalty = sum([int(assignment[z_i]) * int(assignment[z_j]) - int(assignment[z_i]) - int(assignment[z_j]) for z_i, z_j in graph_edges])
    vertex_penalty = sum([int(i) for i in assignment] )
    return -(3*edge_penalty + vertex_penalty)

def max_clique_hamiltonian(assignment, graph_edges):
    graph_edges_complement = nx_graph_from(len(assignment), graph_edges, complement=True).edges
    edge_penalty = sum([int(assignment[z_i]) * int(assignment[z_j]) - int(assignment[z_i]) - int(assignment[z_j]) for z_i, z_j in graph_edges_complement])
    vertex_penalty = sum([int(i) for i in assignment] )
    return -(3*edge_penalty + vertex_penalty)

