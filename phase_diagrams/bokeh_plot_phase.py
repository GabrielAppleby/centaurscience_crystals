import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, GlyphRenderer, Slider, Select, ColorBar
from bokeh.palettes import Viridis256, Category10, Category20
from bokeh.plotting import figure, DEFAULT_TOOLS, Figure
from bokeh.transform import factor_cmap, linear_cmap

NEIGHBORS = list(range(2, 20))
COLORING = ['atomic_group', 'hdbscan']

X_AXIS_TEMPLATE: str = '{}_x'
Y_AXIS_TEMPLATE: str = '{}_y'

HOVER_TOOLTIP_FORMAT: str = """
          <div>
            <hr>
              <p>Name: @name</p>
              <p>Atomic Group: @atomic_group</p>
              <p>HDB Cluster: @hdbscan</p>
            <hr>
          </div>
        """

df: pd.DataFrame = pd.read_csv('crystal_phase_projections.csv', index_col=False,
                               dtype={'atomic_group': str, 'hdbscan': str})
df = df.sort_values('atomic_group')

neighbor_slider: Slider = Slider(title="N_Neighbors", start=NEIGHBORS[0], end=NEIGHBORS[-1],
                                 value=NEIGHBORS[0], step=1)
color_select: Select = Select(title="Color", value='atomic_group', options=COLORING)

source: ColumnDataSource = ColumnDataSource(data=dict(x=[],
                                                      y=[],
                                                      name=[],
                                                      label=[],
                                                      atomic_group=[],
                                                      hdbscan=[]))

mapper = factor_cmap(field_name='label',
                     palette=Category10[4],
                     factors=sorted(df['atomic_group'].unique()))

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


def update():
    neighbor = neighbor_slider.value
    color = color_select.value

    if color == 'atomic_group':
        test = factor_cmap(field_name='label',
                           palette=Category10[4],
                           factors=sorted(df['atomic_group'].unique()))
    else:
        test = factor_cmap(field_name='label',
                           palette=Category20[20],
                           factors=sorted(df['hdbscan'].unique()))

    scatter.glyph.fill_color = test
    source.data = dict(
        x=df[X_AXIS_TEMPLATE.format(neighbor)],
        y=df[Y_AXIS_TEMPLATE.format(neighbor)],
        name=df['name'],
        label=df[color],
        atomic_group=df['atomic_group'],
        hdbscan=df['hdbscan']
    )


controls = [neighbor_slider, color_select]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = row(*controls, width=200, sizing_mode='scale_width')
layout = column(column(inputs, p, sizing_mode='stretch_both'), sizing_mode='stretch_both')
update()

curdoc().add_root(layout)
