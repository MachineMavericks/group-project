import networkx as nx

class NXGraph(nx.Graph):
    def __init__(self, graph, dataset_number=1, weight=None):
        super().__init__()
        if dataset_number == 1:
            if weight == "passage":
                for node in graph.nodes:
                    unique_days = list(set([int(passage.get_day()) for passage in node.get_passages()]))
                    weight = sum([int(passage.get_stay_time()) for passage in node.get_passages()])
                    self.add_node(node_for_adding=node.get_id(),
                                  weight=weight,
                                  lat=node.get_position().get_lat(),
                                  lon=node.get_position().get_lon(),
                                  working_days=unique_days)
                for edge in graph.edges:
                    unique_days = list(set([int(travel.get_day()) for travel in edge.get_travels()]))
                    weight = sum([int(travel.get_travel_time()) for travel in edge.get_travels()])
                    self.add_edge(u_of_edge=edge.get_fromNode(),
                                  v_of_edge=edge.get_destNode(),
                                  weight=weight,
                                  working_days=unique_days)
            elif weight == "trains":
                # TODO: IMPLEMENT THIS.
                pass

        elif dataset_number == 2:
            # TODO: IMPLEMENT THIS.
            pass

def main():
    pass
if __name__ == "__main__":
    main()
