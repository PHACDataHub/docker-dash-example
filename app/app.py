import plotly.graph_objects as go  # or plotly.express as px
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import itertools
import sys
from helpers import *
from callbacks import get_callbacks, df, column_descs

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'CSFI metrics correlation dashboard'

server = app.server

get_callbacks(app)

app.layout = html.Div([
    dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Customer Service Feedback Initiative (CSFI) metrics dashboard: User's guide")),
                dbc.ModalBody([
                    html.P("The purpose of this dashboard is to explore what performance metrics (e.g. # of incidents opened) affect customer satisfaction."),
                    html.P("The intended audience is the CSFI team and SSC data analysts familiar with the source datasets."),
                    html.P([
                        html.B('Explore the data by clicking the colored squares in the correlation matrix.'),
                        html.Span(' The matrix may take a few seconds to load.')
                    ]),
                    html.P("The data is derived from several tables in the Enterprise Data Repository (EDR), with metrics transformed to align with CSFI scores."),
                    html.P([
                            "To explore the data in another tool, download ",
                            html.A('csfi_metrics.csv', href='./assets/csfi_metrics.csv'),
                            " and ",
                            html.A('csfi_metrics_column_descriptions.csv', href='./assets/csfi_metrics_column_descriptions.csv'),
                    ]),
                    html.P(["Please note that this is an early prototype not intended for distribution. There is no French translation and the interactive plots are not keyboard accessible. ",
                            "Contact the CTOB ",
                            html.A('Data Science and Artificial Intelligence unit (DSAI)', href='mailto:ssc.dsai-sdia.spc@canada.ca?subject=CSFI metrics dashboard prototype'),
                            " for more information or access to the GitHub repository."]),

                    dbc.Collapse([
                    html.H2('Correlation matrix'),

                    html.P('The columns of the correlation matrix (left panel) represent the different scores reported through General and Service CSFI surveys. \
                            The rows represent SSC performance metrics collected from the EDR. Note that these are aggregated overall, not grouped by service. \
                            (In DSAI\'s prior exploration of the data, grouping by individual services did not reveal any new insights.)'),
                    html.P('A dark blue square indicates a strong positive correlation (as metric increases, so does score). \
                            A dark red square indicates a strong negative correlation (as metric decreases, score increases). \
                            A less saturated or white square indicates a weak correlation.'),
                    
                    html.H3('How correlation is calculated'),
                    html.P('There are 3 factors contributing to the color of the square:'),
                    html.Ol([
                        html.Li('The overall Pearson correlation, where all data points are considered.'),
                        html.Li('The per-survey Pearson correlation, where correlation is performed for each survey period and averaged.'),
                        html.Li('The number of data points supporting the correlation. A higher number of data points gives us more confidence that the correlation is valid.')
                    ]),
                    html.P('Based on experimentation, we have given the per-survey correlation twice the weight of the overall correlation. \
                        The weighted average is then scaled by a sigmoid function on the number of data points. To improve readability, the values are then scaled to occupy more of the color range (-1 to +1).'),

                    html.H3('Filters and options'),
                    html.P('Click "Show filters" to select specific departments, data sources or surveys of interest.'),
                    # html.P('This area also includes options for the correlation matrix. "Correlate per-survey period" calculates the correlations in a different way, \
                    #         and has a large effect on the outcome. Here, changes over time for each department are ignored, and the correlation captures only \
                    #         the differences between departments within particular survey periods. After calculating correlation for each survey period, the correlations \
                    #         are averaged.'),
                    # html.P('Note that calculating correlation per-survey-period greatly reduces the number of points available for correlation. Generally you should lower the minimum points \
                    #         slider in this case. This also allows metrics which have only recently been tracked (e.g. ONYX) to be shown in the correlation matrix.'),
                    html.P('The "Add department-normalized metrics" checkbox scales metrics across departments of different sizes, which would otherwise \
                            be difficult to correlate due to the variance. First, it calculates the average for the metric per-department across all selected survey periods. \
                            Then, the metric for each survey period is divided by the departmental average. This emphasizes changes over time, rather than an absolute \
                            comparison between departments. The resultant metrics are suffixed with "_nrml". Note that this is applied only to metrics that record totals, not averages.'),
                    html.P('The correlation matrix shows at most 100 rows. By using the dropdown "Sort rows", you may bring different metrics into view.'),

                    html.H2('Details panel'),

                    html.P("Clicking a square in the correlation matrix will display the relevant details in the right pane."),

                    html.H3('Line chart'),
                    html.P("The line chart shows change over time for both the selected metric and CSFI score, averaged across all departments. \
                            Generally speaking, CSFI scores have increased over time, so the yellow score line has a positive slope (the line is higher on the right side). \
                            When a metric positive correlates with a score (blue square in the correlation matrix), the green metric line will usually have a positive slope. \
                            This indicates that as the metric increases, the score also increases. In contrast, when the lines have opposite slopes, \
                            this usually reflects a negative correlation."),
                    html.P("However, the average across departments does not tell the whole story. This only reflects change over time, and does not compare between departments \
                            within individual survey periods."),

                    html.H3('Scatter plot'),
                    html.P('Complementing the line plot, the scatter plot shows the relationship between score and metric for individual data points (per department, per survey), \
                            without displaying the time component. The line of best fit captures both changes over time and comparison of departments within a survey period. \
                            As with the green line in the line plot, a negative slope indicates that as the metric decreases, the score increases (and vice-versa for a positive slope). \
                            Hover over the line to inspect the R squared (that indicates confidence in the correlation).'),
                    html.P(['Use the "Group scatterplot" dropdown to isolate changes over time per-department, or compare exclusively within individual survey periods.']),
                    html.P('Grouping by department will generate a line of best fit for each department, ignoring the differences between departments in a particular survey. \
                            This may lead you to a different conclusion than the averages suggest. \
                            Some departments may not follow the overall trend. Again, it can be useful to inspect the R squared number for each line. \
                            Due to the small number of data points when grouping by department (at most, the number of surveys) these correlations are less indicative of general patterns.'),
                    html.P('Grouping by survey date is extremely valuable, as it exclusively compares departmental scores / metrics within the same survey period, \
                            ignoring changes over time. Here, the individual lines of best fit are for each survey \
                            and the slopes may be very different (often flat, or a mix of positive and negative slopes). Depending on the metric, this may be a more accurate \
                            view of the metric\'s effect. When the same pattern emerges on average (ungrouped), when grouped per department, and when grouped per survey date \
                            - and has a high R squared value - you can be more confident that the metric has a stronger relationship to the CSFI score.'),

                    html.H3('Scatter animation'),
                    html.P('The scatter animation - which must be enabled through a checkbox - shows all the available information for a metric/score combination in one visual. \
                            This is accomplished by animating the time component. Each frame of the animation displays the scatterplot for a particular survey period, \
                            including the average across departments (the larger point). To make it easier to track the movement of points, they have different colors and shapes. \
                            Although a trend line is not displayed for each frame, you can intuit a correlation within a survey when the points generally fall on a sloped line. \
                            Correlation over time is demonstrated through the animation. If the points generally move upward and to the right, there is a positive correlation over time. \
                            Alternately, when the points move upward and to the left, there is a negative correlation over time.'),
                    html.P('A note on "fly-ins": When points suddenly emerge from outside the frame (the bottom right corner), these departments did not have sufficient data for the previous \
                            survey periods. And when points don\'t move between surveys, it may reflect either that the department has reported the exact same values, or more likely, that \
                            the department skipped a survey and thus their score is unchanged.'),
                    html.P('Generating the animations takes around 1 second, so be patient when this option is enabled. (This is why it is off by default.)'),

                    html.H3('Metric and score details'),
                    html.P('At the bottom of the right panel, details about the metric and score are given, including the data sources.'),
                    html.P('Note the number of data points correlated. \
                            A very low number of points (< 40), or non-zero points, should reduce your confidence in the correlation. With a low number of points, outliers can have a large \
                            effect on the Pearson correlation, and lines of best fit.'),

                    html.H2('Interpreting the results'),

                    html.P("While the dashboard presents a useful comparison of metrics and satisfaction scores, it is intended to be used for inspiration to identify metrics of interest. \
                            Domain knowledge and further investigation is required to determine if a metric \
                            has a causal, or merely correlative, effect on customer satisfaction."),
                    # html.P("The metric 'M_Date_Int' has been included to demonstrate that correlation does not imply causation. At first glance, this metric may appear to be \
                    #         the main driver of customer satisfaction - i.e. as time passes, CSFI scores will continue to increase. However, it is more likely that there are several other \
                    #         factors, which have changed over time, that are driving the increase in customer satisfaction."),
                    html.P('Note that outliers may be driving the correlation. Looking at the scatterplot, there may be a single point that has a metric 100x larger than the mean. \
                            To mitigate the effect of outliers, group the plots by department, hover over the outlier points to identify the departments, \
                            and then try excluding the outlier departments via the filters.'),
                        ],
                        id="userguide-collapse",
                        is_open=False,
                    ),
                    dbc.Button(
                        "Read more",
                        id="userguide-readmore-button",
                        className="mb-3",
                        color="secondary",
                        n_clicks=0,
                        style={}
                    ),
                ]),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="guide-modal",
            size='xl',
            is_open=True,
        ),
    html.Div([
        html.Header([
            html.H1([
                # html.Img(src='./assets/ssc-logo.png', width='32px', alt=''),
                'CSFI metrics dashboard',
                html.Span([
                    'Click a square in the correlation matrix to explore. ',
                    html.A('User\'s guide', href='#', id='learn-more', n_clicks=0)
                ], id='subheader'),
            ]),
            html.Div([
                dbc.Button(
                    "Show filters",
                    id="collapse-button",
                    className="mb-3",
                    color="primary",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody([
                        html.H3('Survey filters'),
                        html.Label('Survey type'),
                        dcc.Dropdown(
                            ['General', 'Service'],
                            ['General', 'Service'],
                            multi=True,
                            id='survey-type-dropdown'
                        ),
                        html.Label('Survey date'),
                        # dcc.RangeSlider(
                        #     min=df['QUES_PERIOD_END'].dt.year.min(),
                        #     max=df['QUES_PERIOD_END'].dt.year.max(),
                        #     marks={str(year): str(year)
                        #            for year in df['QUES_PERIOD_END'].dt.year.unique()},
                        #     value=[df['QUES_PERIOD_END'].dt.year.min(
                        #     ), df['QUES_PERIOD_END'].dt.year.max()],
                        #     id='survey-year-slider',
                        # ),
                        dcc.Dropdown(
                            multi=True,
                            id='survey-date-dropdown'
                        ),
                        html.Label('Department'),
                        dcc.Dropdown(
                            sorted(df['GC_ORG_ACRONYM_EN'].unique()),
                            sorted(df['GC_ORG_ACRONYM_EN'].unique()),
                            multi=True,
                            id='department-dropdown'
                        ),
                        html.Label('Data source'),
                        dcc.Dropdown(
                            sorted(list(set([c.split('_')[1] for c in df.columns if c.startswith('M_') and c != 'M_Date_Int']))),
                            sorted(list(set([c.split('_')[1] for c in df.columns if c.startswith('M_') and c != 'M_Date_Int']))),
                            multi=True,
                            id='data-source-dropdown'
                        ),
                        html.H3('Correlation options'),
                        dcc.Checklist(
                            [
                                # 'Correlate per-survey period (ignores changes over time)',
                                'Add department-normalized metrics'
                            ],
                            [],
                            id='corr-options-checklist'
                        ),
                        html.Div([
                            html.Label('Minimum aggregated data points'),
                            dcc.Slider(
                                min=0,
                                max=100,
                                value=35,
                                id='min-records-slider',
                            ),
                        ], style={'display': 'none'}),
                        html.Label('Sort columns'),
                        dcc.Dropdown(
                            ['Standard order', 'Sum', 'Absolute sum', 'Max value', 'Absolute max value'],
                            'Standard order',
                            id='corr-sort-columns-dropdown'
                        ),
                        html.Label('Sort rows'),
                        dcc.Dropdown(
                            ['Alphabetical', 'Sum', 'Absolute sum', 'Negative sum', 'Max value', 'Absolute max value'] + (
                                list(itertools.chain(*zip(['Absolute ' + c for c in df.columns if c.startswith('S_')], [c for c in df.columns if c.startswith('S_')])))),
                            'Absolute sum',
                            id='corr-sort-rows-dropdown'
                        ),
                    ],
                        id='filters'
                    )),
                    id="filters-collapse",
                    is_open=False,
                ),
            ]),
        ]), html.Div(
            dcc.Graph(
                id='heatmap',
                clickData={'points': [{
                    'x': [c for c in df.columns if c.startswith('S_')][0],
                    'y': [c for c in df.columns if c.startswith('M_')][0],
                    'z': None
                    }]},
            ),
        ),
    ],
    id='left-panel'),
    html.Div([
        html.Button(type="button", className="btn-close", id='rpanel-close-btn', n_clicks=0, title='Close'),
        html.Div(
            [
                html.Div([
                    html.Label('Line plot displays'),
                    dcc.Dropdown(
                            ['Metric and score', 'Metric per dept', 'Score per dept'],
                            'Metric and score',
                            multi=False,
                            id='lineplot-options-dropdown'
                        ),
                ]),
                html.Div([
                    html.Label('Group scatterplot'),
                    dcc.Dropdown(
                            ['None', 'Department', 'Survey date'],
                            'None',
                            multi=False,
                            id='scatter-options-dropdown'
                        ),
                ]),
                dcc.Checklist(
                    ['Scatter histograms', 'Scatter animation (slow)'],
                    [],
                    id='scatter-options-checklist',
                    # style={'display': 'none'}
            ),
            ], id='right-checklists'),
            html.Div([
                html.Div(dcc.Graph(id='avg-time'), style={}),
                html.Div(dcc.Graph(id='scatter'), style={}),
                html.Div([], id='animated-scatter'),
                html.Div([], id='metric-details'),
            ], id='right-details'
        ),
    ],
    id='right-panel'
    ),
])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8050", debug=True)
