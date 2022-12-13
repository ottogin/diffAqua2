import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def subsample_point_cloud(point_cloud, n_subsample, fields=None):
    n_subsample = min(n_subsample, point_cloud.shape[0])
    indices = np.random.choice(point_cloud.shape[0], n_subsample)

    fields_sub = None
    if fields is not None:
        fields_sub = fields[indices]
    return point_cloud[indices], fields_sub

def show(traces):
    layout = go.Layout(scene_aspectmode='data')
    fig = go.Figure(layout=layout)
    for tr in traces:
        fig.add_trace(tr)

    fig.update_layout(template="plotly_dark")
    fig.show()

def show_grid(*plots, shape=None, title="", names=None):
    if shape is None:
        shape = (len(plots), 1)
        
    if names is None:
        names = []
    fig = make_subplots(cols=shape[0],  rows=shape[1],
                        specs=[[{'type': 'surface'}] * shape[0]] * shape[1],
                        subplot_titles=names)

    for idx, pl in enumerate(plots):
        for tr in pl:
            fig.add_trace(tr, col=idx % shape[0] + 1, row=idx // shape[0] + 1)
    
    fig.update_scenes(aspectmode='data')
    fig.update_layout(template="plotly_dark")
    fig.show()

def plot_3d_mesh(verts, faces):
    trace = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], 
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], )
    return [trace]

def plot_3d_point_cloud(points, n_subsample=None, fields=None):
    if n_subsample:
        points_sub, fields_sub = subsample_point_cloud(points, n_subsample, fields=fields)
    else:
        points_sub = points

    if fields is not None:
        trace = go.Scatter3d(x=points_sub[:, 0], y=points_sub[:, 1], z=points_sub[:, 2],
                             mode='markers', marker = dict(size=2, color=fields_sub))
    else:
        trace = go.Scatter3d(x=points_sub[:, 0], y=points_sub[:, 1], z=points_sub[:, 2],
                             mode='markers', marker = dict(size=2))
    return [trace]