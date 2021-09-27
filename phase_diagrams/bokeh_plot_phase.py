import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, GlyphRenderer, Slider
from bokeh.plotting import figure, DEFAULT_TOOLS, Figure
from bokeh.transform import factor_cmap

NEIGHBORS = list(range(2, 20))

X_AXIS_TEMPLATE: str = '{}_x'
Y_AXIS_TEMPLATE: str = '{}_y'

HOVER_TOOLTIP_FORMAT: str = """
          <div>
            <hr>
              <p>Name: MP-@name</p>
              <p>PT Group: @atomic_group</p>
            <hr>
          </div>
        """

df: pd.DataFrame = pd.read_csv('crystal_phase_projections.csv', index_col=False,
                               dtype={'atomic_group': str})
df = df.sort_values('atomic_group')

neighbor_slider: Slider = Slider(title="N_Neighbors", start=NEIGHBORS[0], end=NEIGHBORS[-1],
                                 value=NEIGHBORS[0], step=1)

source: ColumnDataSource = ColumnDataSource(data=dict(x=[],
                                                      y=[],
                                                      name=[],
                                                      atomic_group=[]))

p: Figure = figure(tools=[DEFAULT_TOOLS] + [HoverTool(tooltips=HOVER_TOOLTIP_FORMAT)],
                   sizing_mode='stretch_both',
                   output_backend="webgl")

scatter: GlyphRenderer = p.scatter('x',
                                   'y',
                                   line_width=.1,
                                   size=5,
                                   line_color='white',
                                   legend='atomic_group',
                                   fill_color=factor_cmap('atomic_group', 'Category10_4',
                                                          df['atomic_group'].unique()),
                                   source=source)


def update():
    neighbor = neighbor_slider.value

    source.data = dict(
        x=df[X_AXIS_TEMPLATE.format(neighbor)],
        y=df[Y_AXIS_TEMPLATE.format(neighbor)],
        name=df['name'],
        atomic_group=df['atomic_group']
    )


controls = [neighbor_slider]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = row(*controls, width=200, sizing_mode='scale_width')
layout = column(column(inputs, p, sizing_mode='stretch_both'), sizing_mode='stretch_both')
update()

curdoc().add_root(layout)
