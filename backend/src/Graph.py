# import networkx as nx
import pandas as pd

def read_mtx(filepath):
    # Read mtx file, and store each line in a list
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Remove the lines that start with '%'
    lines = [line for line in lines if not line.startswith('%')]
    # Store the lines in a 2D list
    lines = [line.split() for line in lines]
    return lines

class Passage:
    def __init__(self, train, arr_time, dep_time, stay_time, date):
        self.train = train
        self.arr_time = arr_time
        self.dep_time = dep_time
        self.stay_time = stay_time
        self.date = date

class Position:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

class Node:
    def __init__(self, id, position, passages):
        self.id = id
        self.position = position
        self.passages = passages

class Edge:
    def __init__(self, id, fromNode, destNode, mileage, duration):
        self.id = id
        self.fromNode = fromNode
        self.destNode = destNode
        self.mileage = mileage
        self.duration = duration

class Graph:
    def __init__(self, filepath):
        # READ THE FILE:
        self.filepath = filepath
        self.df = pd.read_csv(filepath, low_memory=False)
        self.nodes = []

        # NODES CONSTRUCTION:
        for st_id in self.df['st_id'].unique():
            lat = self.df[self.df['st_id'] == st_id]['lat'].values[0]
            lon = self.df[self.df['st_id'] == st_id]['lon'].values[0]
            position = Position(lat, lon)
            node = Node(st_id, position, [])
            self.nodes.append(node)
        self.nodes = sorted(self.nodes, key=lambda node: node.id)
        self.nodesAmount = len(self.nodes)

        # EDGES CONSTRUCTION:
        self.edges = []
        self.trains = [train for train in self.df['train'].unique()]
        for index, row in self.df.iterrows():
            if (not (row['stay_time'] == '-' and self.df.iloc[index - 1]['stay_time'] == '-') and (row['train'] == self.df.iloc[index - 1]['train'])):
                fromNode = Node(self.df.iloc[index - 1]['st_id'], Position(self.df.iloc[index - 1]['lat'], self.df.iloc[index - 1]['lon']), [])
                destNode = Node(row['st_id'], Position(row['lat'], row['lon']), [])
                edge = Edge(index, fromNode, destNode, row['mileage'], 0)
                self.edges.append(edge)
        self.edgesAmount = len(self.edges)

        # PASSAGES CONSTRUCTION:
        self.df = self.df.sort_values(by=['st_id'])
        # For each node, add the passages:
        index = 0
        st_id = self.df.iloc[index]['st_id']
        for row in self.df.itertuples():
            if row.st_id != st_id:
                index += 1
                st_id = row.st_id
            passage = Passage(row.train, row.arr_time, row.dep_time, row.stay_time, row.date)
            self.nodes[index].passages.append(passage)


    def print_attributes(self):
         print("GRAPH.NODES(len=" + str(len(self.nodes)) + ")={")
         print("Node={id=" + str(self.nodes[0].id) + ", lat=" + str(self.nodes[0].position.lat) + ", lon=" + str(
             self.nodes[0].position.lon) + ", ", end='')
         print("Passages[Node=" + str(self.nodes[0].id) + "]={", end='')
         for passage in self.nodes[0].passages:
             print("Passage={train=" + str(passage.train) + ", arr_time=" + str(passage.arr_time) + ", dep_time=" + str(
                 passage.dep_time) + ", stay_time=" + str(passage.stay_time) + ", date=" + str(passage.date) + "}, ",
                   end='')
         print("}")
         print("...")
         print("Node={id=" + str(self.nodes[len(self.nodes) - 1].id) + ", lat=" + str(
             self.nodes[len(self.nodes) - 1].position.lat) + ", lon=" + str(
             self.nodes[len(self.nodes) - 1].position.lon) + ", ", end='')
         print("Passages[Node=" + str(self.nodes[len(self.nodes) - 1].id) + "]={", end='')
         for passage in self.nodes[len(self.nodes) - 1].passages:
             print("Passage={train=" + str(passage.train) + ", arr_time=" + str(passage.arr_time) + ", dep_time=" + str(
                 passage.dep_time) + ", stay_time=" + str(passage.stay_time) + ", date=" + str(passage.date) + "}, ",
                   end='')
         print("}")
         print("}")
         print("GRAPH.EDGES(len=" + str(len(self.edges)) + ")={")
         print("Edge={id=" + str(self.edges[0].id) + ", fromNode=" + str(self.edges[0].fromNode.id) + ", destNode=" + str(
             self.edges[0].destNode.id) + ", mileae=" + str(self.edges[0].mileage) + ", duration=" + str(self.edges[0].duration) + "}, ", end='')
         print("}")
         print("...")
         print("Edge={id=" + str(self.edges[len(self.edges) - 1].id) + ", fromNode=" + str(
             self.edges[len(self.edges) - 1].fromNode.id) + ", destNode=" + str(
             self.edges[len(self.edges) - 1].destNode.id) + ", mileage=" + str(
             self.edges[len(self.edges) - 1].mileage) + ", duration=" + str(
             self.edges[len(self.edges) - 1].duration) + "}, ", end='')
         print("}")

def main():
    graph = Graph('data/input/railway.csv')
    graph.print_attributes()
if __name__ == '__main__':
    main()