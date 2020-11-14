import os

import igraph as ig
import plotly as py
import plotly.graph_objs as go

from trade import dataset

this_file_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(this_file_path, "..", "..", "data", "dataset_raw")

data = dataset.TradeDemoYearByYearDataset(dataset_path)

year_idx = 0

demo = data[year_idx].x[:, 2].tolist()

edge_index = data[year_idx].edge_index
edge_attr = data[year_idx].edge_attr

node_to_colo = {}
colo_node_iter = 0
colonial_edges = []
for edge_idx in range(edge_attr.shape[0]):
    if edge_attr[edge_idx, 5] == 1:
        new_edge = tuple(edge_index[:, edge_idx].tolist())

        if new_edge[0] not in node_to_colo:
            node_to_colo[new_edge[0]] = colo_node_iter
            colo_node_iter += 1
        if new_edge[1] not in node_to_colo:
            node_to_colo[new_edge[1]] = colo_node_iter
            colo_node_iter += 1

        colonial_edges.append(tuple([node_to_colo[new_edge[0]], node_to_colo[new_edge[1]]]))

N = colo_node_iter

G=ig.Graph(colonial_edges, directed=True)
layt=G.layout(layout='kk')

Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
Yn=[layt[k][1] for k in range(N)]# y-coordinates
Xe=[]
Ye=[]
for e in colonial_edges:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]



trace1=go.Scatter(x=Xe,
               y=Ye,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=1),
               hoverinfo='none'
               )

trace2=go.Scatter(x=Xn,
               y=Yn,
               mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=6,
                             color=demo,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=1)
                             ),
               hoverinfo='text'
               )

axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout = go.Layout(
         title="Network of trade relationships between countries in 1945",
         width=1000,
         height=1000,
         showlegend=False,
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text="Data source: <a href='https://www.v-dem.net/en/'>VDem dataset</a>",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ],    )

data=[trace1, trace2]
fig=go.Figure(data=data, layout=layout)

fig.show()