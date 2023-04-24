import networkx as nx
from src.Models.NXGraph import NXGraph
from src.Services.NXGraphService import largest_connected_component_ratio, global_efficiency_ratio
from src.Draft.ResilienceDraft import average_shortest_path_length_ratio_2
from tqdm import tqdm

def main():
    ch_pickle_path = 'static/output/chinese.pickle'
    in_pickle_path = 'static/output/indian.pickle'
    pickles = [ch_pickle_path, in_pickle_path]
    fractions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    metrics = ['degree_centrality', 'betweenness_centrality']

    for pickle_path in pickles:
        csv_string = 'pickle_path,metric,fraction,lcc,global_efficiency\n'
        for metric in metrics:
            for fraction in tqdm(fractions):
                nxgraph = NXGraph(pickle_path=pickle_path, dataset_number=1, day=None)
                nx_graph_copy = nxgraph.copy()
                nodes_to_remove = int(len(nx_graph_copy) * fraction)
                metric_dict = {}
                if metric == 'degree_centrality':
                    metric_dict = nx.degree_centrality(nx_graph_copy)
                    nx.set_node_attributes(nx_graph_copy, metric_dict, 'degree_centrality')
                    nx.set_node_attributes(nxgraph, metric_dict, 'degree_centrality')
                elif metric == 'betweenness_centrality':
                    metric_dict = nx.betweenness_centrality(nx_graph_copy)
                    nx.set_node_attributes(nx_graph_copy, metric_dict, 'betweenness_centrality')
                    nx.set_node_attributes(nxgraph, metric_dict, 'betweenness_centrality')

                sorted_metric_dict = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:nodes_to_remove]
                for node, degree in sorted_metric_dict:
                    nx_graph_copy.remove_node(node)

                # evaluate
                # lcc:
                lcc = largest_connected_component_ratio(nxgraph, nx_graph_copy)
                # global_eff:
                global_eff = global_efficiency_ratio(nxgraph, nx_graph_copy)
                # sp:
                # avgsp = average_shortest_path_length_ratio_2(nxgraph, nx_graph_copy)
                # add to csv:
                csv_string += f'{pickle_path},{metric},{fraction},{lcc},{global_eff}\n'
        with open(f'static/output/{pickle_path.split("/")[-1].split(".")[0]}_resilience.csv', 'w') as f:
            f.write(csv_string)

if __name__ == '__main__':
    main()

