import networkx as nx
import json

# NXGRAPH LOADE: GRAPH -> NETWORKX
def nxgraph_loader(graph):
    gnx = nx.Graph()
    for node in graph.nodes:
        gnx.add_nodes_from([(node.id, {'lat': node.position.lat, 'lon': node.position.lon})])
    for edge in graph.edges:
        gnx.add_edges_from([(edge.fromNode.id, edge.destNode.id,
                             {'fromLat': edge.fromNode.position.lat, 'fromLon': edge.fromNode.position.lon,
                              'destLat': edge.destNode.position.lat, 'destLon': edge.destNode.position.lon})])
    # BUG FIX: CONVERT INTS TO STRINGS:
    gnx = nx.convert_node_labels_to_integers(gnx, first_label=0, ordering='default', label_attribute=None)
    for edge in gnx.edges:
        for attr in gnx.edges[edge]:
            gnx.edges[edge][attr] = str(gnx.edges[edge][attr])
    return gnx


# CONVERTER: NETWORKX <-> JSON  (! IMPORTANT: This format will lose the edge attributes)
def nx_to_json_string(gnx):
    adj_list = nx.to_dict_of_lists(gnx)
    return adj_list


def nx_to_json_file(gnx, filepath):
    json = nx_to_json_string(gnx)
    with open(filepath, 'w') as outfile:
        json.dump(json, outfile)


def json_string_to_nx(adj_list):
    gnx = nx.from_dict_of_lists(adj_list)
    return gnx


def json_file_to_nx(filepath):
    with open(filepath) as json_file:
        adj_list = json.load(json_file)
    gnx = json_string_to_nx(adj_list)
    return gnx


# CONVERTER: NETWORKX <-> GML
def nx_to_gml_string(gnx):
    string = nx.generate_gml(gnx)
    return string


def nx_to_gml_file(gnx, filepath):
    nx.write_gml(gnx, filepath)


def gml_string_to_nx(string):
    gnx = nx.parse_gml(string)
    return gnx


def gml_file_to_nx(filepath):
    gnx = nx.read_gml(filepath)
    return gnx


# CONVERTER: NETWORKX <-> GEXF  (! IMPORTANT: This format can't be converted to string)
def nx_to_gexf_file(gnx, filepath):
    nx.write_gexf(gnx, filepath)


def gexf_file_to_nx(filepath):
    gnx = nx.read_gexf(filepath)
    return gnx


def print_metrics(nxgraph):
    # Number of nodes:
    print('Number of nodes in the NetworkX graph: ' + str(nxgraph.number_of_nodes()))
    # Number of edges:
    print('Number of edges in the NetworkX graph: ' + str(nxgraph.number_of_edges()))
    # Average degree:
    print('Average degree of the NetworkX graph: ' + str(
        sum(dict(nxgraph.degree()).values()) / len(dict(nxgraph.degree()).values())))
    # Average clustering coefficient:
    print('Average clustering coefficient of the NetworkX graph: ' + str(nx.average_clustering(nxgraph)))
    # Average shortest path length:
    temp_graph = nxgraph.copy()
    subgraph = temp_graph.subgraph(max(nx.connected_components(nxgraph), key=len))
    print("Average shortest path length: " + str(nx.average_shortest_path_length(subgraph)))
    # Diameter:
    print("Diameter: " + str(nx.diameter(subgraph)))
    # Radius:
    print("Radius: " + str(nx.radius(subgraph)))
    # Center:
    print("Center: " + str(nx.center(subgraph)))
    # Periphery:
    print("Periphery: " + str(nx.periphery(subgraph)))
