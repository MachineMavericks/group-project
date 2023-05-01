from pathlib import Path

from backend.src.Services.NXGraphService import plotly_default
from backend.src.Services.NXGraphService import plotly_heatmap
from backend.src.Services.NXGraphService import plotly_resilience
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
import pickle


def test_case_1():
    # Valid input
    pickle_path = 'static/output/chinese.pickle'
    plotly_default(pickle_path=pickle_path)


def test_case_2():
    # Valid input with day and output path
    pickle_path = 'static/output/chinese.pickle'
    day = 10
    output_path = 'static/output/chinese.pickle'
    plotly_default(pickle_path=pickle_path, day=day, output_path=output_path)


def test_case_3():
    # Invalid input (file not found)
    pickle_path = 'static/output/chinese.pickle'
    plotly_default(pickle_path=pickle_path)

def test_case_4():
    pickle_path = 'static/output/chinese.pickle'
    plotly_heatmap(pickle_path)

def test_case_5():
    pickle_path = 'static/output/chinese.pickle'
    plotly_heatmap(pickle_path, component="node", metric="total_minutes", day=1)

def test_case_6():
    pickle_path = 'static/output/chinese.pickle'
    plotly_heatmap(pickle_path, component="edge", metric="total_travels")

def test_case_7():
    pickle_path = 'static/output/chinese.pickle'
    plotly_heatmap(pickle_path, component="node", metric="invalid_metric")


def test_case_8():
    pickle_path = 'static/output/chinese.pickle'
    plotly_heatmap(pickle_path, component="edge", metric="invalid_metric")

def test_case_9():
    pickle_path = 'static/output/chinese.pickle'
    plotly_resilience(pickle_path, component=None)

def test_case_10():
    pickle_path = 'static/output/chinese.pickle'
    plotly_resilience(pickle_path, component="node", strategy="targeted", metric="degree_centrality",
                      fraction=None)

def test_case_11():
    pickle_path = 'static/output/chinese.pickle'
    plotly_resilience(pickle_path, component="node", strategy="targeted", metric="degree_centrality",
                      fraction="0.5", output_path="path/to/output.html")

def test_case_12():
    pickle_path = 'static/output/chinese.pickle'
    plotly_resilience(pickle_path, component="edge", strategy="targeted")

def test_case_13():
    pickle_path = 'static/output/chinese.pickle'
    plotly_resilience(pickle_path, component="node", strategy="random")

def test_case_14():
    pickle_path = 'static/output/chinese.pickle'
    plotly_resilience(pickle_path, component="edge", strategy="random")



# Run all test cases
if __name__ == '__main__':
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    test_case_7()
    test_case_8()
    test_case_9()
    test_case_10()
    test_case_11()
    test_case_12()
    test_case_13()
    test_case_14()