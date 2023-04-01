import networkx as nx

class NXGraph(nx.Graph):
    def __init__(self, graph, dataset_number=1, weight=None):
        super().__init__()
        if dataset_number == 1:
            if weight == "passage":
                for node in graph.nodes:
                    weight = sum([int(passage.get_stay_time()) for passage in node.get_passages()])
                    self.add_node(node.get_id(), weight=weight, lat=node.get_position().get_lat(),
                                     lon=node.get_position().get_lon())
                for edge in graph.edges:
                    weight = sum([int(travel.get_travel_time()) for travel in edge.get_travels()])
                    self.add_edge(edge.get_fromNode(), edge.get_destNode(), weight=weight)
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
