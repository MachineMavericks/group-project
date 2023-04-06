import networkx as nx
import pickle
import time


class NXGraph(nx.MultiGraph):
    def __init__(self, graph=None, pickle_path=None, dataset_number=1, day=None, save_gml=None, output_path=None):
        start = time.time()
        super().__init__()
        if graph is not None or pickle_path is not None:
            if pickle_path is not None:
                print("Constructing the NXGraph using the pickle file...")
                graph = pickle.load(open(pickle_path, "rb"))
            elif graph is not None:
                print("Constructing the NXGraph using the Graph object...")
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
                            total_travels=len(travels),
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
                            total_passages=len(passages),
                            total_minutes=total_minutes,
                            working_days=list(set([int(passage.get_day()) for passage in node.get_passages()]))
                        )
            elif dataset_number == 2:
                # TODO: IMPLEMENT THIS.
                pass
            if save_gml and output_path is not None:
                nx.write_gml(self, output_path, stringizer=str)
            print("NXGraph constructed in {} seconds.".format(round(time.time() - start, 2)))


def main():
    pass


if __name__ == "__main__":
    main()
