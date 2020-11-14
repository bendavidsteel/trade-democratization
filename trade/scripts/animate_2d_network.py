import json
import os

import igraph as ig
import pandas as pd
import plotly.graph_objects as go
import torch

from trade import dataset

this_file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(this_file_path, "..", "..", "data")
dataset_path = os.path.join(data_path, "dataset_raw")

data = dataset.TradeDemoYearByYearDataset(dataset_path)

node_dict_path = os.path.join(data_path, "dataset", "processed", "node_dict.pt")
node_dict = torch.load(node_dict_path)

with open(os.path.join(data_path, "country_mapping.json"), "r") as f:
    country_mapping = json.loads(f.read())

with open(os.path.join(data_path, "country_names.json"), "r") as f:
    country_names = json.loads(f.read())

N = len(country_mapping)

years = [year_dict["year"] for year_dict in node_dict]

# invert all node mappings
node_mappings = []
for year_dict in node_dict:
    reverse_mapping = {}
    for mapping in year_dict["node_mapping"].items():
        reverse_mapping[mapping[1]] = mapping[0]
    node_mappings.append(reverse_mapping)

# make figure
fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
}

# fill in most of layout
fig_dict["layout"]["xaxis"] = {
    "range": [-10, 10], 
    'showgrid': False, 
    'zeroline': False, 
    'visible': False
    }
fig_dict["layout"]["yaxis"] = {
    "range": [-10, 10],
    'showgrid': False, 
    'zeroline': False, 
    'visible': False
    }
fig_dict['layout']['showlegend'] = False
fig_dict['layout']['annotations'] = [
    {
        "showarrow": False,
        "text": "Data source: <a href='https://www.v-dem.net/en/'>VDem dataset</a>",
        "xref": 'paper',
        "yref": 'paper',
        "x": 0,
        "y": 0.02,
        "xanchor": 'left',
        "yanchor": 'bottom',
        "font": {
            "size": 14
        }
    },
    {
        "showarrow": False,
        "text": "Data source: <a href='http://www.cepii.fr/CEPII/En/bdd_modele/presentation.asp?id=32'>TradHist dataset</a>",
        "xref": 'paper',
        "yref": 'paper',
        "x": 0,
        "y": 0.04,
        "xanchor": 'left',
        "yanchor": 'bottom',
        "font": {
            "size": 14
        }
    }
]
fig_dict["layout"]["hovermode"] = "closest"
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 1000, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 300,
                                                                    "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
]

sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Year:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}

# make data
year_idx = 0

demo = data[year_idx].x[:, 2].tolist()

edge_index = data[year_idx].edge_index
edge_attr = data[year_idx].edge_attr

node_to_cou = node_mappings[year_idx]
cou_to_colo = {}
colo_to_cou = {}
colo_node_iter = 0
colonial_edges = []
for edge_idx in range(edge_attr.shape[0]):
    if edge_attr[edge_idx, 5] == 1:
        new_edge = tuple(edge_index[:, edge_idx].tolist())

        if node_to_cou[new_edge[0]] not in cou_to_colo:
            cou_to_colo[node_to_cou[new_edge[0]]] = colo_node_iter
            colo_to_cou[colo_node_iter] = node_to_cou[new_edge[0]]
            colo_node_iter += 1
        if node_to_cou[new_edge[1]] not in cou_to_colo:
            cou_to_colo[node_to_cou[new_edge[1]]] = colo_node_iter
            colo_to_cou[colo_node_iter] = node_to_cou[new_edge[1]]
            colo_node_iter += 1

        colonial_edges.append(tuple([cou_to_colo[node_to_cou[new_edge[0]]], cou_to_colo[node_to_cou[new_edge[1]]]]))

N = colo_node_iter

G=ig.Graph(colonial_edges, directed=True)
layt=G.layout(layout='kk')

Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
Yn=[layt[k][1] for k in range(N)]# y-coordinates
nIds = [str(colo_to_cou[k]) for k in range(N)]
nTexts = [country_names[country_mapping[colo_to_cou[k]][0]] for k in range(N)]
nDemo = [(demo[node_dict[year_idx]["node_mapping"][colo_to_cou[k]]] * 10) + 10 for k in range(N)]
Xe=[]
Ye=[]
eIds = []
for e in colonial_edges:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    eIds += str([colo_to_cou[e[0]], colo_to_cou[e[1]]].sort())

trace1_dict = {
    "x": Xe,
    "y": Ye,
    "ids": eIds,
    "mode": 'lines',
    "line": { "color": 'rgb(125,125,125)', "width": 1},
    "hoverinfo": 'none'
}

trace2_dict = {
    "x": Xn,
    "y": Yn,
    "ids": nIds,
    "mode": 'markers',
    "name": 'Countries',
    "marker": {
        "symbol": 'circle',
        "size": nDemo,
        "color": nDemo,
        "colorscale": 'Viridis',
        "line": { "color": 'rgb(50,50,50)', "width": 1 }
    },
    "hoverinfo": 'text',
    "hovertext": nTexts
}

fig_dict["data"].append(trace1_dict)
fig_dict["data"].append(trace2_dict)

# make frames
for year_idx in range(len(years)):
    frame = {"data": [], "name": years[year_idx]}
    demo = data[year_idx].x[:, 2].tolist()

    edge_index = data[year_idx].edge_index
    edge_attr = data[year_idx].edge_attr

    node_to_cou = node_mappings[year_idx]
    cou_to_colo = {}
    colo_to_cou = {}
    colo_node_iter = 0
    colonial_edges = []
    for edge_idx in range(edge_attr.shape[0]):
        if edge_attr[edge_idx, 5] == 1:
            new_edge = tuple(edge_index[:, edge_idx].tolist())

            if node_to_cou[new_edge[0]] not in cou_to_colo:
                cou_to_colo[node_to_cou[new_edge[0]]] = colo_node_iter
                colo_to_cou[colo_node_iter] = node_to_cou[new_edge[0]]
                colo_node_iter += 1
            if node_to_cou[new_edge[1]] not in cou_to_colo:
                cou_to_colo[node_to_cou[new_edge[1]]] = colo_node_iter
                colo_to_cou[colo_node_iter] = node_to_cou[new_edge[1]]
                colo_node_iter += 1

            colonial_edges.append(tuple([cou_to_colo[node_to_cou[new_edge[0]]], cou_to_colo[node_to_cou[new_edge[1]]]]))

    N = colo_node_iter

    G=ig.Graph(colonial_edges, directed=True)
    layt=G.layout(layout='kk')

    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    nIds = [str(colo_to_cou[k]) for k in range(N)]
    nTexts = [country_names[country_mapping[colo_to_cou[k]][0]] for k in range(N)]
    nDemo = [(demo[node_dict[year_idx]["node_mapping"][colo_to_cou[k]]] * 10) + 10 for k in range(N)]
    Xe=[]
    Ye=[]
    eIds = []
    for e in colonial_edges:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
        eIds += str([colo_to_cou[e[0]], colo_to_cou[e[1]]].sort())

    trace1_dict = {
        "x": Xe,
        "y": Ye,
        "ids": eIds,
        "mode": 'lines',
        "line": { "color": 'rgb(125,125,125)', "width": 1},
        "hoverinfo": 'none'
    }

    trace2_dict = {
        "x": Xn,
        "y": Yn,
        "ids": nIds,
        "mode": 'markers',
        "name": 'Countries',
        "marker": {
            "symbol": 'circle',
            "size": nDemo,
            "color": nDemo,
            "colorscale": 'Viridis',
            "line": { "color": 'rgb(50,50,50)', "width": 1 }
        },
        "hoverinfo": 'text',
        "hovertext": nTexts
    }

    frame["data"].append(trace1_dict)
    frame["data"].append(trace2_dict)

    fig_dict["frames"].append(frame)
    slider_step = {"args": [
        [years[year_idx]],
        {"frame": {"duration": 300, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 300}}
    ],
        "label": years[year_idx],
        "method": "animate"}
    sliders_dict["steps"].append(slider_step)


fig_dict["layout"]["sliders"] = [sliders_dict]

fig = go.Figure(fig_dict)

#fig.show()
fig.write_html(os.path.join(this_file_path, "colonization_visual.html"))