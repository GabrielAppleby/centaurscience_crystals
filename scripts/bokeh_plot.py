from pathlib import Path
from typing import List

import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, ColorBar, HoverTool, GlyphRenderer
from bokeh.palettes import Viridis256, Category10, Category20
from bokeh.plotting import figure, DEFAULT_TOOLS, Figure
from bokeh.transform import linear_cmap, factor_cmap

METRICS: List[str] = ['euclidean',
                      'cosine']
NEIGHBORS = ['2', '4', '5', '10', '20', '40', '80']

EMBEDDINGS = ['g2vec_degree',
              'g2vec_atomic_number',
              'g2vec_atomic_group',
              'fgsd_degree',
              'gl2vec_degree']

COLORING = ['formation_energy', 'space_group', 'gl2vec_degree_euclidean_cluster','gl2vec_degree_cosine_cluster','g2vec_atomic_group_euclidean_cluster','g2vec_atomic_group_cosine_cluster','g2vec_atomic_number_euclidean_cluster','g2vec_atomic_number_cosine_cluster','g2vec_degree_euclidean_cluster','g2vec_degree_cosine_cluster','fgsd_degree_euclidean_cluster','fgsd_degree_cosine_cluster']

FIGURE_NAME_TEMPLATE: str = '{}_{}.html'

X_AXIS_TEMPLATE: str = '{}_{}_{}_x'
Y_AXIS_TEMPLATE: str = '{}_{}_{}_y'

HOVER_TOOLTIP_FORMAT: str = """
          <div>
            <hr>
              <p>ID: MP-@cid</p>
              <p>Formula: @formula</p>
              <p>Space Group: @space_group</p>
              <p>PT Group: @atomic_group</p>
              <p>FE: @fe</p>
            <hr>
          </div>
        """

df: pd.DataFrame = pd.read_csv('crystal_graph_embedding_projections.csv', index_col=False,
                               dtype={'space_group': str,
                                      'most_prevalent_atomic_group': str})

embedding_select: Select = Select(title="Embedding:", value=EMBEDDINGS[0], options=EMBEDDINGS)
metric_select: Select = Select(title="Metric:", value=METRICS[0], options=METRICS)
neighbor_select: Select = Select(title="N_Neighbors", value=NEIGHBORS[0], options=NEIGHBORS)
color_select: Select = Select(title="Color", value='formation_energy', options=COLORING)

source: ColumnDataSource = ColumnDataSource(data=dict(x=[],
                                                      y=[],
                                                      label=[],
                                                      space_group=[],
                                                      fe=[],
                                                      formula=[],
                                                      atomic_group=[],
                                                      avg_bond_dist=[]))

mapper = linear_cmap(field_name='label',
                     palette=Viridis256,
                     low=min(df['formation_energy']),
                     high=max(df['formation_energy']))

p: Figure = figure(tools=[DEFAULT_TOOLS] + [HoverTool(tooltips=HOVER_TOOLTIP_FORMAT)],
                   sizing_mode='stretch_both',
                   output_backend="webgl")

scatter: GlyphRenderer = p.scatter('x',
                                   'y',
                                   fill_color=mapper,
                                   line_width=.1,
                                   size=5,
                                   line_color='white',
                                   legend='label',
                                   source=source)

color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
p.add_layout(color_bar, 'right')


def update():
    embedding = embedding_select.value
    metric = metric_select.value
    neighbor = neighbor_select.value
    color = color_select.value
    if color == 'formation_energy':
        p.legend.visible = False
        low = min(df['formation_energy'])
        high = max(df['formation_energy'])
        test = linear_cmap(field_name='label',
                           palette=Viridis256,
                           low=low,
                           high=high)
        mapper['transform'].low = low
        mapper['transform'].high = high
    elif color == 'most_prevalent_atomic_group':
        color_bar.visible = False
        test = factor_cmap(field_name='label',
                           palette=Category20[17],
                           factors=sorted(df['most_prevalent_atomic_group'].unique()))
    elif color == 'space_group':
        color_bar.visible = False
        test = factor_cmap(field_name='label',
                           palette=Category10[7],
                           factors=sorted(df['space_group'].unique()))
    else:
        p.legend.visible = False
        low = min(df[color])
        high = max(df[color])
        test = linear_cmap(field_name='label',
                           palette=Viridis256,
                           low=low,
                           high=high)
        mapper['transform'].low = low
        mapper['transform'].high = high
    # print(len(df[color].unique()))
    scatter.glyph.fill_color = test
    source.data = dict(
        x=df[X_AXIS_TEMPLATE.format(embedding, metric, neighbor)],
        y=df[Y_AXIS_TEMPLATE.format(embedding, metric, neighbor)],
        label=df[color],
        cid=df['cids'],
        space_group=df['space_group'],
        fe=df['formation_energy'],
        formula=df['formula'],
        atomic_group=df['most_prevalent_atomic_group']
    )

    if color == 'formation_energy' or color.endswith('cluster'):
        color_bar.visible = True
    else:
        p.legend.visible = True


controls = [embedding_select, metric_select, neighbor_select, color_select]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = row(*controls, width=200, sizing_mode='scale_width')
layout = column(column(inputs, p, sizing_mode='stretch_both'), sizing_mode='stretch_both')
update()

curdoc().add_root(layout)
