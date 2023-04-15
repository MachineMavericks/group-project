# IMPORTS=
import time as time
import numpy as np
import warnings
import os
import pickle
from tqdm import tqdm

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
    def __init__(self, filepath, output_dir=None, save_csvs=False, save_pickle=False):
        self.filename = filepath.split('/')[-1].split('.')[0]
        # CHECK IF THE GRAPH IS ALREADY CONSTRUCTED (ALREADY EXISTING .PICKLE FILE):
        if os.path.isfile(output_dir + self.filename + '.pickle'):
            print("Found already existing " + self.filename + ".pickle file, aborted redundant graph construction.")
        # OTHERWISE, CONSTRUCT THE GRAPH:
        else:
            # START TIMER:
            warnings.simplefilter(action='ignore', category=FutureWarning)
            start_time = time.time()

            # READ THE FILE:
            self.filepath = filepath
            if self.filename == 'indian':
                self.df = indian_railway_preprocessing_pipeline(pd.read_csv(filepath, low_memory=False), save_csv=save_csvs, output_path=((output_dir + 'indian_pp.csv') if save_csvs and (output_dir is not None) else None))\
                    if not os.path.isfile(output_dir + 'indian_pp.csv') else pd.read_csv(output_dir + 'indian_pp.csv', low_memory=False)
            elif self.filename == 'chinese':
                self.df = chinese_railway_preprocessing_pipeline(pd.read_csv(filepath, low_memory=False), save_csv=save_csvs, output_path=((output_dir + 'chinese_pp.csv') if save_csvs and (output_dir is not None) else None))\
                    if not os.path.isfile(output_dir + 'chinese_pp.csv') else pd.read_csv(output_dir + 'chinese_pp.csv', low_memory=False)
            else:
                raise Exception("Invalid filename, please use 'indian.csv' or 'chinese.csv' railway datasets.")
            print("Constructing the Graph using the preprocessed data...")

            # NODES PREPROCESSING:
            print("Found already existing " + self.filename + "_npp.csv file, aborted redundant nodes preprocessing .") if os.path.isfile(output_dir + self.filename + '_npp.csv') else None
            df_ = nodes_preprocessing(self.df,
                                      save_csv=save_csvs,
                                      output_path=((output_dir + self.filename + '_npp.csv') if save_csvs and (output_dir is not None) else None))\
                if not os.path.isfile(output_dir + self.filename + '_npp.csv')\
                else pd.read_csv(output_dir + self.filename + '_npp.csv', low_memory=False)
            # NODES CONSTRUCTION:
            print("Constructing the nodes...")
            nodes_df = df_.copy()\
                .groupby(['st_id', 'lat', 'lon']).size().reset_index(name='count')\
                .sort_values(by=['st_id'], ascending=True)\
                .reset_index(drop=True)
            nodes_passages_df = df_.copy()
            # self._nodes = []
            self._nodes = {}
            for index, row in tqdm(nodes_df.iterrows(), total=len(nodes_df)):
                node_passages_df = nodes_passages_df[(nodes_passages_df['st_id'] == row['st_id'])] \
                    .sort_values(by=['arr_time'], ascending=True) \
                    .reset_index(drop=True)
                nodes_passages = []
                for index_, row_ in node_passages_df.iterrows():
                    nodes_passages.append(NodePassage(row_['train'], int(row_['day']), row_['arr_time'], int(row_['stay_time'])))
                # self._nodes.append(Node(row['st_id'], Position(row['lat'], row['lon']), nodes_passages))
                self._nodes[row['st_id']] = Node(row['st_id'], Position(row['lat'], row['lon']), nodes_passages)
            print("Nodes constructed.")

            # EDGES PREPROCESSING:
            print("Found already existing " + self.filename + "_epp.csv file, aborted redundant edges preprocessing .") if os.path.isfile(output_dir + self.filename + '_epp.csv') else None
            df_ = edges_preprocessing(self.df,
                                      save_csv=save_csvs,
                                      output_path=((output_dir + self.filename + '_epp.csv') if save_csvs and (output_dir is not None) else None))\
                if not os.path.isfile(output_dir + self.filename + '_epp.csv')\
                else pd.read_csv(output_dir + self.filename + '_epp.csv', low_memory=False)
            # EDGES CONSTRUCTION:
            print("Constructing the edges...")
            edges_df = df_.copy() \
                .groupby(['dep_st_id', 'arr_st_id', 'mileage']).size().reset_index(name='count') \
                .sort_values(by=['dep_st_id', 'arr_st_id', 'mileage'], ascending=True) \
                .reset_index(drop=True)
            edge_travels_df = df_.copy()
            # self._edges = []
            self._edges = {}
            for index, row in tqdm(edges_df.iterrows(), total=len(edges_df)):
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
                # fromNode = self.get_node_by_id(row['dep_st_id'])
                fromNode = self.get_nodes()[row['dep_st_id']]
                # destNode = self.get_node_by_id(row['arr_st_id'])
                destNode = self.get_nodes()[row['arr_st_id']]
                edge = Edge(index, fromNode, destNode, int(row['mileage']), travels)
                self.get_edges()[fromNode.get_id(), destNode.get_id()].append(edge) if (fromNode.get_id(), destNode.get_id()) in self.get_edges() else self.get_edges().update({(fromNode.get_id(), destNode.get_id()): [edge]})
            print("Edges constructed.")

            # END.
            print("Graph constructed in " + str(np.round((time.time() - start_time), 2)) + " seconds.")

            # SAVE THE GRAPH:
            if save_pickle and (output_dir is not None):
                with open(output_dir + self.filename+".pickle", 'wb') as f:
                    pickle.dump(self, f)
                print("Graph saved in " + output_dir + self.filename + ".pickle")

    # GETTERS:
    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges


def main():
    pass


if __name__ == "__main__":
    main()
