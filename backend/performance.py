from src.Services.NXGraphService import *
from src.Models.NXGraph import NXGraph
from src.Models.Graph import Graph

def performance():
    times = "dataset, task, time\n"
    filepaths = ['resources/data/input/chinese.pickle', 'resources/data/input/indian.pickle']
    pickles = ['resources/data/output/chinese.pickle', 'resources/data/output/indian.pickle']
    for filepath in filepaths:
        start = time.time()
        graph = Graph(filepath=filepath, output_dir="resources/data/output/")
        times += f'{filepath}, graph, {time.time() - start}\n'
        print(times)

        pickle = pickles[filepaths.index(filepath)]
        start = time.time()
        nxgraph = NXGraph(pickle_path=pickle, dataset_number=1)
        times += f'{filepath}, nxgraph, {time.time() - start}\n'
        print(times)

        start = time.time()
        plotly_default(pickle_path=pickle, day=1)
        times += f'{filepath}, plotly_default, {time.time() - start}\n'
        print(times)

        start = time.time()
        plotly_heatmap(pickle_path=pickle, component="node", metric="degree_centrality", day=None, output_path=None)
        times += f'{filepath}, plotly_heatmap, {time.time() - start}\n'
        print(times)

        start = time.time()
        plotly_resilience(pickle_path=pickle, day=None, strategy="targeted", component="node", metric="degree_centrality", fraction="0.01", output_path=None, smallworld=None)
        times += f'{filepath}, plotly_resilience, {time.time() - start}\n'
        print(times)

        start = time.time()
        plotly_clustering(pickle_path=pickle, algorithm="Euclidian k-mean", day=None, output_path=None)
        times += f'{filepath}, plotly_clustering_kmeans, {time.time() - start}\n'
        print(times)

        start = time.time()
        plotly_clustering(pickle_path=pickle, algorithm="Louvain", weight="total_minutes", day=None, output_path=None)
        times += f'{filepath}, plotly_clustering_louvain, {time.time() - start}\n'
        print(times)

        start = time.time()
        plotly_small_world(pickle_path=pickle, day=None, output_path="static/output/temp2.html")
        times += f'{filepath}, plotly_small_world, {time.time() - start}\n'
        print(times)

        # Write to file
        with open("resources/data/output/performance.csv", "w") as f:
            f.write(times)



def main():
    performance()


if __name__ == '__main__':
    main()
