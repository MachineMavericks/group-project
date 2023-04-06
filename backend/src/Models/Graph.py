# IMPORTS=
import time as time
import numpy as np
import warnings

# MODELS=
from src.Models.EdgeTravel import EdgeTravel
from src.Models.Edge import Edge
from src.Models.Node import Node
from src.Models.NodePassage import NodePassage
from src.Models.Position import Position
# SERVICES=
from src.Preprocessing.DataPreprocessing import *
from src.Preprocessing.EdgePreprocessing import *
from src.Preprocessing.NodePreprocessing import *


class Graph:
    def __init__(self, filepath, save_csvs=False, output_dir=None):
        # START TIMER:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        start_time = time.time()

        # READ THE FILE:
        self.filepath = filepath
        self.df = preprocessing_pipeline(pd.read_csv(filepath, low_memory=False), save_csv=save_csvs, path=((output_dir + 'railway.csv') if save_csvs and (output_dir is not None) else None))

        # TODO: NODES CONSTRUCTION:
        df_ = nodes_preprocessing(self.df, save_csv=save_csvs, output_path=((output_dir + 'nodes_pp.csv') if save_csvs and (output_dir is not None) else None))
        nodes_df = df_.copy()\
            .groupby(['st_id', 'lat', 'lon']).size().reset_index(name='count')\
            .sort_values(by=['st_id'], ascending=True)\
            .reset_index(drop=True)
        nodes_passages_df = df_.copy()
        print("Constructing the nodes...")
        self.nodes = []
        for index, row in nodes_df.iterrows():
            node_passages_df = nodes_passages_df[(nodes_passages_df['st_id'] == row['st_id'])] \
                .sort_values(by=['arr_time'], ascending=True) \
                .reset_index(drop=True)
            nodes_passages = []
            for index_, row_ in node_passages_df.iterrows():
                nodes_passages.append(
                    NodePassage(row_['train'], int(row_['day']), row_['arr_time'], int(row_['stay_time'])))
            self.nodes.append(Node(row['st_id'], Position(row['lat'], row['lon']), nodes_passages))
        print("Nodes constructed.")

        # TODO: EDGES CONSTRUCTION:
        df_ = edges_preprocessing(self.df, save_csv=save_csvs, output_path=((output_dir + 'edges_pp.csv') if save_csvs and (output_dir is not None) else None))
        print("Constructing the edges...")
        edges_df = df_.copy() \
            .groupby(['dep_st_id', 'arr_st_id', 'mileage']).size().reset_index(name='count') \
            .sort_values(by=['dep_st_id', 'arr_st_id', 'mileage'], ascending=True) \
            .reset_index(drop=True)
        edge_travels_df = df_.copy()

        self.edges = []
        for index, row in edges_df.iterrows():
            edge_travels_df_ = edge_travels_df[(edge_travels_df['dep_st_id'] == row['dep_st_id'])
                                               & (edge_travels_df['arr_st_id'] == row['arr_st_id'])
                                               & (edge_travels_df['mileage'] == row['mileage'])] \
                .sort_values(by=['dep_date'], ascending=True) \
                .reset_index(drop=True)
            travels = []
            for index_, row_ in edge_travels_df_.iterrows():
                travels.append(
                    EdgeTravel(row_['train_id'], row_['dep_st_id'], int(row_['day']), row_['dep_date'],
                               int(row_['travel_time']),
                               row_['arr_st_id']))
            # Create the edge:
            edge = Edge(index, row['dep_st_id'], row['arr_st_id'], int(row['mileage']), travels)
            self.edges.append(edge)
        print("Edges constructed.")

        # END.
        print("Graph constructed in " + str(np.round((time.time() - start_time), 2)) + " seconds.")

    # GETTERS:
    def get_node_by_id(self, id):
        for node in self.nodes:
            if node.get_id() == id:
                return node
        return None

    def get_edge_by_id(self, id):
        for edge in self.edges:
            if edge.get_id() == id:
                return edge
        return None


def main():
    pass


if __name__ == "__main__":
    main()
