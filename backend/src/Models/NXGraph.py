import networkx as nx

class NXGraph(nx.MultiGraph):
    def __init__(self, graph, dataset_number=1, day=None):
        super().__init__()
        # DATASET N°1 = RAILWAY.CSV
        if dataset_number == 1:
            regarded_days = [day_ for day_ in [1, 2, 3, 4] if day % day_ == 0] if day is not None else [1, 2, 3, 4]
            regarded_nodes = []
            for edge in graph.edges:
                travels = edge.get_travels()
                if day is not None:
                    travels = [travel for travel in travels if travel.get_day() in regarded_days]
                if len(travels) > 0:
                    total_minutes = sum([int(travel.get_travel_time()) for travel in travels])
                    mileage = edge.get_mileage()
                    total_mileage = mileage * len(travels)
                    self.add_edge(
                        u_for_edge=edge.get_fromNode(),
                        v_for_edge=edge.get_destNode(),
                        mileage=mileage,
                        travels_amount=len(travels),
                        total_minutes=total_minutes,
                        total_mileage=total_mileage,
                        working_days=list(set([int(travel.get_day()) for travel in edge.get_travels()]))
                    )
                    if edge.get_fromNode() not in regarded_nodes:
                        regarded_nodes.append(edge.get_fromNode())
                    if edge.get_destNode() not in regarded_nodes:
                        regarded_nodes.append(edge.get_destNode())
            for node in graph.nodes:
                if node.get_id() in regarded_nodes:
                    passages = node.get_passages()
                    if day is not None:
                        passages = [passage for passage in passages if passage.get_day() in regarded_days]
                    total_minutes = sum([int(passage.get_stay_time()) for passage in passages])
                    self.add_node(
                        node_for_adding=node.get_id(),
                        lat=node.get_position().get_lat(),
                        lon=node.get_position().get_lon(),
                        passages_amount=len(passages),
                        total_minutes=total_minutes,
                        working_days=list(set([int(passage.get_day()) for passage in node.get_passages()]))
                    )
            # Find each node degree, betweenness and closeness centrality, and add it as node attributes:
            degree_centrality = nx.degree_centrality(self)
            nx.set_node_attributes(self, degree_centrality, 'degree_centrality')
            betweenness_centrality = nx.betweenness_centrality(self)
            nx.set_node_attributes(self, betweenness_centrality, 'betweenness_centrality')
            closeness_centrality = nx.closeness_centrality(self)
            nx.set_node_attributes(self, closeness_centrality, 'closeness_centrality')
            # Find each edge betweenness centrality, and add it as edge attributes:
            edge_betweenness_centrality = nx.edge_betweenness_centrality(self)
            nx.set_edge_attributes(self, edge_betweenness_centrality, 'betweenness_centrality')

        elif dataset_number == 2:
            # TODO: IMPLEMENT THIS.
            pass

def main():
    pass
if __name__ == "__main__":
    main()
