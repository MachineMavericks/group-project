import heapq
import pickle
from datetime import datetime, timedelta, time, date
import math

import plotly.graph_objects as go
import plotly.express as px

from src.Models.NXGraph import NXGraph
from src.Models.Graph import Graph
import networkx as nx

from src.Services.NXGraphService import df_from_nxgraph, fig_update_layout


def astar_path(graph, start, end):
    """
    Finds the shortest path between the start and end nodes of the given graph using the A* algorithm.

    :param graph: The graph to search.
    :param start: The starting node.
    :param end: The ending node.
    :return: A list of nodes representing the shortest path.
    """
    path = nx.astar_path(graph, start, end, heuristic=lambda u, v: heuristic_function(graph, u, v), weight='mileage')
    return path


def heuristic_function(nx_graph, u, v):
    """
    Heuristic function to estimate the distance between the current node and the target node.
    The function return the inverse of the distance as nodes with higher distances are the worst candidates.

    :param nx_graph: railway network
    :param u: The current node.
    :param v: The target node.
    :return: An estimate of the distance between the two nodes.
    """
    if u == v:
        return 0
    fromLat, fromLon = nx_graph.nodes[u]['lat'], nx_graph.nodes[u]['lon']
    destLat, destLon = nx_graph.nodes[v]['lat'], nx_graph.nodes[v]['lon']

    return 1 / haversine(fromLat, fromLon, destLat, destLon)


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on the Earth's surface using the Haversine formula.

    :param lat1: The latitude of the first point.
    :param lon1: The longitude of the first point.
    :param lat2: The latitude of the second point.
    :param lon2: The longitude of the second point.
    :return: The distance between the two points, in kilometers.
    """
    earth_radius = 6371  # kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c
    return distance


def travel_cost(graph: Graph, current_node, next_node, current_time):
    best_cost = float('inf')
    best_travel = 0
    best_arrival = 0
    # for each edge between current_node and next_node (max is 2)
    for rail in graph.get_edges()[current_node, next_node]:
        # for each travel on the edge
        for travel in rail.get_travels():
            dep_time = datetime.strptime(travel.get_dep_time(), '%H:%M:%S').time()

            if dep_time < current_time:
                dt1 = datetime.combine(date.today(), current_time)
                dt2 = datetime.combine(date.today() + timedelta(days=1), dep_time)
            else:
                dt1 = datetime.combine(date.today(), current_time)
                dt2 = datetime.combine(date.today(), dep_time)

            wait = dt2 - dt1

            cost = round(wait.total_seconds() / 60) + travel.get_travel_time()
            if cost < best_cost:
                best_cost = cost
                best_travel = travel
                best_arrival = dt2 + timedelta(minutes=travel.get_travel_time())

    return best_cost, best_arrival.time(), best_travel.get_train_id()


# TODO: mind that Graph does not consider day ==> if day specified nx_graph and graph will be different

def a_star_shortest_path(nx_graph: NXGraph, graph: Graph, root, goal, departure_time):
    """
    Computes the shortest path from a station to another departing at a chosen time with A* search.
    :param nx_graph: Railway network
    :param graph: original graph object
    :param root: starting node
    :param goal: target node
    :param departure_time: time of departure
    :return: path and expected arrival time
    """

    # Define a priority queue setting the priority as: f(n) = g(n) + h(n), where g(n) is the
    # cost of the path so far and h(n) is the heuristic cost to the goal node.
    frontier = [(0, root)]
    # path reconstruction dict
    came_from = {}

    # Define a dictionary 'record' to keep track of the cost of the best path so far for each node.
    cost_record = {root: (0, departure_time, '_')}

    times, trains = [], []

    # while queue is not empty
    while frontier:
        # Get the node with the lowest f(n) value from the queue --> 'current_node'.
        _, current = heapq.heappop(frontier)
        # If goal node is found ==> return
        if current == goal:
            path = [current]
            while current != root:
                current = came_from[current]
                path.append(current)
                times.append(cost_record[current][1])
                trains.append(cost_record[current][2])
            path.reverse()
            times.reverse()
            trains.reverse()
            return path, times, trains
        # for each neighbor compute cost(root, neighbor)= record[current_node] + travel_cost(current_node, neighbor)
        for neighbor in nx_graph.successors(current):
            current_time = cost_record[current][1]
            cost_to_neighbor = travel_cost(graph, current, neighbor, current_time=current_time)
            cost = cost_record[current][0] + cost_to_neighbor[0]
            # print(current, '-->', neighbor, '  ', current_time, '-->', cost_to_neighbor[1])
            # if neighbor is not in 'record' the dictionary that keeps track best costs so far for each node...
            # ... or if cost(root, neighbor) < record[neighbor]
            if neighbor not in cost_record or cost < cost_record[neighbor][0]:
                # Add/update the cost for neighbor in 'record'
                cost_record[neighbor] = cost, cost_to_neighbor[1], cost_to_neighbor[2]
                # compute priority = cost(root, current) + heuristic(neighbor, goal)
                priority = cost + heuristic_function(nx_graph, neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
                # Add/update train and time to tracking lists

    # if we exit the while loop then goal has not been reached ==> no existing path
    print("No existing path")
    return None


# Usage example
pickle_path = "../../static/output/chinese.pickle"
test_nx_graph = NXGraph(pickle_path=pickle_path, dataset_number=1, day=2)
test_graph = pickle.load(open(pickle_path, "rb"))
start = 15.0
end = 146.0
depart_time1 = time(8, 0, 0)
depart_time2 = time(10, 0, 0)
depart_time3 = time(15, 0, 0)
depart_time4 = time(21, 0, 0)
depart_time5 = time(23, 0, 0)

# path = astar_path(test_nx_graph, start, end)
path1 = a_star_shortest_path(test_nx_graph, test_graph, start, end, depart_time1)
# path2 = a_star_shortest_path(test_nx_graph, test_graph, start, end, depart_time2)
# path3 = a_star_shortest_path(test_nx_graph, test_graph, start, end, depart_time3)
# path4 = a_star_shortest_path(test_nx_graph, test_graph, start, end, depart_time4)
# path5 = a_star_shortest_path(test_nx_graph, test_graph, start, end, depart_time5)

# print('Simple:', path1)
print('Departing at ', depart_time1, ':\n_______________\nStations:')
for t in path1[0]:
    print(t, ' --> ', end='')
print('\nTimes:')
for t in path1[1]:
    print(t, ' --> ', end='')
print('\nTrains:')
for t in path1[2]:
    print(t, ' --> ', end='')

# print('Departing at ', depart_time2, ':', path2)
# print('Departing at ', depart_time3, ':', path3)
# print('Departing at ', depart_time4, ':', path4)
# print('Departing at ', depart_time5, ':', path5)

# c_time = time(19, 30, 0)
# print(travel_cost(test_graph, 2317.0, 1688.0, c_time)[0])

# # NODES DATAFRAME:
# df_nodes = df_from_nxgraph(test_nx_graph, component="node")
# # EDGES DATAFRAME:
# df_edges = df_from_nxgraph(test_nx_graph, component="edge")
#
# # Create the figure with the nodes and edges
# fig = px.scatter_mapbox(df_nodes[df_nodes["Node ID"] == -1], lat="Latitude", lon="Longitude", hover_name="Node ID",
#                         hover_data=["Total passages", "Total minutes"], zoom=3.5, mapbox_style="open-street-map",
#                         height=800, center=dict(lat=36, lon=117))
#
# # Add the edges as disconnected lines in a single trace
# edges_x = []
# edges_y = []
# for edge in test_nx_graph.edges(data=True):
#     x0 = edge[2]['fromLon']
#     y0 = edge[2]['fromLat']
#     x1 = edge[2]['destLon']
#     y1 = edge[2]['destLat']
#     edges_x.append(x0)
#     edges_x.append(x1)
#     edges_x.append(None)
#     edges_y.append(y0)
#     edges_y.append(y1)
#     edges_y.append(None)
# fig.add_trace(go.Scattermapbox(
#     lat=edges_y,
#     lon=edges_x,
#     mode='lines',
#     line=dict(width=1, color='grey'),
#     hoverinfo='text',
#     hovertext="Edge from " + df_edges['Source Node'].astype(str) + " to " + df_edges['Destination Node'].astype(
#         str) + "<br>Total travels: " + df_edges['Total travels'].astype(str) + "<br>Total minutes: " + df_edges[
#                   'Total minutes'].astype(str),
#     name="Edges"
# ))
#
# # Add the nodes as markers
# fig.add_scattermapbox(
#     lat=df_nodes['Latitude'],
#     lon=df_nodes['Longitude'],
#     mode='markers',
#     marker=dict(size=5, color='red'),
#     hoverinfo='text',
#     hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
#     hovertext="Node n°" + df_nodes['Node ID'].astype(str) + "<br>Position: (Lat=" + df_nodes['Latitude'].astype(
#         str) + ", Lon=" + df_nodes['Longitude'].astype(str) + ")<br>Total passages: " + df_nodes[
#                   'Total passages'].astype(str) + "<br>Total minutes: " + df_nodes['Total minutes'].astype(str),
#     name="Nodes"
# )
#
# # Add the two paths as separate traces
# for i, path in enumerate([path1, path2]):
#     path_edges_x = []
#     path_edges_y = []
#     for j in range(len(path) - 1):
#         source = path[j]
#         target = path[j + 1]
#         source_data = df_nodes[df_nodes['Node ID'] == source].iloc[0]
#         target_data = df_nodes[df_nodes['Node ID'] == target].iloc[0]
#         path_edges_x += [source_data['Longitude'], target_data['Longitude'], None]
#         path_edges_y += [source_data['Latitude'], target_data['Latitude'], None]
#     fig.add_trace(go.Scattermapbox(
#         lat=path_edges_y,
#         lon=path_edges_x,
#         mode='lines',
#         line=dict(width=2, color='blue' if i == 0 else 'green'),
#         hoverinfo='skip',
#         name="Path " + str(i + 1)
#     ))
#
# # Update the figure layout
# fig_update_layout(fig)
#
# # Show the plot
# fig.show()
