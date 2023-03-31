import networkx as nx
import json

# NXGRAPH LOADE: GRAPH -> NETWORKX
def nxgraph_loader(graph):
    nxgraph = nx.Graph()
    for node in graph.nodes:
        passages = node.get_passages()
        weight = sum([int(passage.get_stay_time()) for passage in passages])
        nxgraph.add_node(node.get_id(), weight=weight, lat=node.get_position().get_lat(),
                         lon=node.get_position().get_lon())
    for edge in graph.edges:
        travels = edge.get_travels()
        weight = sum([int(travel.get_travel_time()) for travel in edge.get_travels()])
        nxgraph.add_edge(edge.get_fromNode(), edge.get_destNode(), weight=weight)
    # BUG FIX: CONVERT INTS TO STRINGS: #TODO: REPAIR
    # nxgraph = nx.convert_node_labels_to_integers(nxgraph, first_label=0, ordering='default', label_attribute=None)
    # for edge in nxgraph.edges:
    #     for attr in nxgraph.edges[edge]:
    #         nxgraph.edges[edge][attr] = str(nxgraph.edges[edge][attr])
    # return nxgraph

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
    print('Average degree of the NetworkX graph: ' + str(sum(dict(nxgraph.degree()).values()) / len(dict(nxgraph.degree()).values())))
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
