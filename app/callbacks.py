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

# Setup
global df
global column_descs
df = pd.read_csv('./assets/csfi_metrics.csv')
df = df.drop(columns=[c for c in df.columns if '_nrml' in c])
df['QUES_PERIOD_END'] = pd.to_datetime(df['QUES_PERIOD_END'])
df['SURVEY_TYPE'] = 'Service'
df.loc[df['S_SRVC_AVG'].isna(), 'SURVEY_TYPE'] = 'General'
df['M_Date_Int'] = df['QUES_PERIOD_END'].apply(lambda x: x.value)
column_descs = pd.read_csv('./assets/csfi_metrics_column_descriptions.csv')

def get_callbacks(app):
    @app.callback(
        Output('metric-details', 'children'),
        Input('heatmap', 'clickData'),
        Input('survey-date-dropdown', 'value'),
        Input('survey-type-dropdown', 'value'),
        Input('department-dropdown', 'value'),
    )
    def metric_details(clickData, survey_dates, survey_types, depts):
        filtered_df = filter_df(df, survey_dates, depts, survey_types)

        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']

        elements = [] # HTML to return

        num_points = len(filtered_df[~(filtered_df[x].isna()) & ~(filtered_df[y].isna())])
        non_zero_points = num_points - len(filtered_df[~(filtered_df[x].isna()) & (filtered_df[y] == 0)])

        corr_level = clickData['points'][0]['z']
        if corr_level is not None:
            elements += [html.P(
                html.B(f'Scaled correlation: {corr_level:.2f}', style={'color': 'darkred' if corr_level < 0 else 'darkblue'})
                )]

        elements += [
            html.P(f'{num_points} points to correlate; {non_zero_points} non-zero points'),
            html.H3('Metric: ' + y, id='metric-name')
        ]

        explanations = {
            'nrml': '(Divided by department average over selected survey periods)',
            'perEmpl': '(Divided by number of employees in department at time of survey)'
        }

        y_clean = y.replace('_nrml', '').replace('_perEmpl', '')
        metric_desc = column_descs.query(f'col == "{y_clean}"')['desc_only'].dropna(how='all')
        metric_src = column_descs.query(f'col == "{y_clean}"')['source'].dropna(how='all')
        if len(metric_desc) > 0:
            metric_desc = metric_desc.values[0]
            nrml = 'nrml' in y
            per_empl = 'perEmpl' in y
            elements.append(html.P(f'{metric_desc} {explanations["nrml"] if nrml else ""} {explanations["perEmpl"] if per_empl else ""}'))

        if len(metric_src) > 0:
            metric_src = metric_src.values[0]
            elements.append(html.P(f'Source: {metric_src}'))

        elements += [
            # html.P(f'Minimum: {filtered_df[y].dropna().min()}; Maximum: {filtered_df[y].max()}'),
            html.H3('Score: ' + x, id='score-name'),
        ]

        score_desc = column_descs.query(f'col == "{x}"')['desc_only'].dropna()
        score_src = column_descs.query(f'col == "{x}"')['source'].dropna()
        if len(score_desc) > 0:
            elements.append(html.P(f'{score_desc.values[0]}'))
        if len(score_src) > 0:
            elements.append(html.P(f'Source: {score_src.values[0]}'))
        
        elements += [
            # html.P(f'Minimum: {filtered_df[x].dropna().min()}; Maximum: {filtered_df[x].max()}'),
        ]

        return elements

    @app.callback(
        Output('survey-date-dropdown', 'options'),
        Output('survey-date-dropdown', 'value'),
        Input('survey-type-dropdown', 'value')
    )
    def filter_dates(survey_types):
        valid_dates = [str(date)[:10] for date in df.loc[df['SURVEY_TYPE'].isin(survey_types), 'QUES_PERIOD_END'].sort_values().unique()]
        return valid_dates, valid_dates


    @app.callback(
        Output('heatmap', 'figure'),
        Input('survey-date-dropdown', 'value'),
        Input('survey-type-dropdown', 'value'),
        Input('department-dropdown', 'value'),
        Input('data-source-dropdown', 'value'),
        Input('min-records-slider', 'value'),
        Input('corr-sort-columns-dropdown', 'value'),
        Input('corr-sort-rows-dropdown', 'value'),
        Input('corr-options-checklist', 'value')
    )
    def update_heatmap(survey_dates, survey_types, depts, data_srcs, min_records, sort_cols, sort_rows, checklist):

        # corr_per_survey = 'Correlate per-survey period (ignores changes over time)' in checklist
        add_norm = 'Add department-normalized metrics' in checklist

        filtered_df = filter_df(df, survey_dates, depts, survey_types, add_norms=add_norm)

        good_cols = [c for c in filtered_df.columns if c.split(
            '_')[1] in data_srcs or not c.startswith('M_') or c.startswith('M_Date')]
        filtered_df = filtered_df[good_cols]

        corr_overall = csfi_corr(filtered_df, min_periods=10, method='pearson', mode='overall')
        corr_per_survey = csfi_corr(filtered_df, min_periods=10, method='pearson', mode='per_survey')

        nonzero_count = count_nonzero_points(filtered_df)

        # Sigmoid function because "enough" points are "enough". We don't really care if there are 200 or 388.
        # conf is arbitrary but increasing it means that the threshold for how many points is "enough" effectively goes down
        # This will produce an output between 0.5 and 1
        conf = 5
        nonzero_scaler = 1 / (1 + np.exp(((nonzero_count / nonzero_count.max().max())) * -conf))
        # Scale to between 0 and 1
        nonzero_scaler = nonzero_scaler * 2 - 1

        corr = ((corr_overall + corr_per_survey) / 2) * nonzero_scaler

        corr = (corr / corr.abs().max().max()).dropna(axis=1, how='all').dropna(axis=0, how='all')

        # Sorting the columns (scores)
        if 'max' in sort_cols.lower() and 'abs' in sort_cols.lower():
            sum_cols = corr.abs().max(axis=0)
        elif 'sum' in sort_cols.lower() and 'abs' in sort_cols.lower():
            sum_cols = corr.abs().sum(axis=0)
        elif 'max' in sort_cols.lower():
            sum_cols = corr.max(axis=0)
        elif 'sum' in sort_cols.lower():
            sum_cols = corr.sum(axis=0)
        elif 'standard' not in sort_cols.lower():
            # Sort based on correlation with a metric
            sum_cols = corr.loc[sort_cols]

        # Sorting the rows (metrics)
        if 'max' in sort_rows.lower() and 'abs' in sort_rows.lower():
            sum_rows = corr.abs().max(axis=1)
        elif 'sum' in sort_rows.lower() and 'abs' in sort_rows.lower():
            sum_rows = corr.abs().sum(axis=1)
        elif 'max' in sort_rows.lower():
            sum_rows = corr.max(axis=1)
        elif 'sum' in sort_rows.lower():
            sum_rows = corr.sum(axis=1)
        elif 'abs' in sort_rows.lower():
            # Sort based on correlation with a score
            sum_rows = corr[sort_rows.replace('Absolute ', '')].abs()
        elif 'alpha' in sort_rows.lower():
            sum_rows = corr.index
        else:
            sum_rows = corr[sort_rows]

        if 'standard' in sort_cols.lower():
            pass
        else:
            corr = corr[sum_cols.sort_values(ascending=False).index]
        if 'alpha' in sort_rows.lower():
            corr = corr.loc[corr.index.sort_values()]
        else:
            corr = corr.loc[sum_rows.sort_values(ascending='negative' in sort_rows.lower()).index]
        
        maxitems = 100
        width, height = calc_heatmap_size(corr, maxitems)

        fig = csfi_corr_plot(corr, title='', maxitems=maxitems, width=width, height=height)
        return fig


    @app.callback(
        Output('avg-time', 'figure'),
        Input('survey-date-dropdown', 'value'),
        Input('survey-type-dropdown', 'value'),
        Input('department-dropdown', 'value'),
        Input('heatmap', 'clickData'),
        Input('lineplot-options-dropdown', 'value')
    )
    def update_avgtime(survey_dates, survey_types, depts, clickData, graph_options):
        filtered_df = filter_df(df, survey_dates, depts, survey_types)

        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']

        if 'Metric per dept' == graph_options:
            # Show the metrics change over time only (not score)
            avg_df = filtered_df.groupby(['QUES_PERIOD_END', 'GC_ORG_ACRONYM_EN'])[
                y].mean().reset_index().dropna()
            fig = px.line(avg_df, x='QUES_PERIOD_END',
                        y=y, color='GC_ORG_ACRONYM_EN', height=280)
            fig.update_layout(showlegend=False)

        elif 'Score per dept' == graph_options:
            # Show the metrics change over time only (not score)
            avg_df = filtered_df.groupby(['QUES_PERIOD_END', 'GC_ORG_ACRONYM_EN'])[
                x].mean().reset_index().dropna()
            fig = px.line(avg_df, x='QUES_PERIOD_END',
                        y=x, color='GC_ORG_ACRONYM_EN', height=280)
            fig.update_layout(showlegend=False)

        else:
            avg_df = filtered_df.groupby(['QUES_PERIOD_END'])[
                    y].mean().reset_index().dropna()
            avg_df2 = filtered_df.groupby(['QUES_PERIOD_END'])[
                    x].mean().reset_index().dropna()

            # Create figure with secondary y-axis (one for metric, one for score)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.update_layout(height=280)
            # Add traces
            fig.add_trace(
                go.Scatter(x=avg_df['QUES_PERIOD_END'], y=avg_df[y], name=f'{y} (all depts avg)', line=dict(color='ForestGreen')),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(x=avg_df2['QUES_PERIOD_END'], y=avg_df2[x], name=f'{x} (all depts avg)', line=dict(color='Goldenrod')),
                secondary_y=False,
            )

            fig.update_yaxes(title_text=y, secondary_y=True)
            fig.update_yaxes(title_text=x, secondary_y=False)

            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
            )

        # Set x-axis title
        fig.update_xaxes(title_text="Survey Period End Date")

        # No margins
        fig.update_layout(
            margin=dict(
                t=80,
                b=0,
                l=0,
                r=0
            )
        )

        fig.update_layout(plot_bgcolor='white',
        #                   paper_bgcolor=bgcolor, font_color='white'
                        )
        fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)', linecolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)', linecolor='rgba(0,0,0,0.1)')
        fig.update_traces(marker=dict(line=dict(width=1,
                                                color='black')),
                        selector=dict(mode='markers'))

        return fig


    @app.callback(
        Output('scatter', 'figure'),
        Input('survey-date-dropdown', 'value'),
        Input('survey-type-dropdown', 'value'),
        Input('department-dropdown', 'value'),
        Input('heatmap', 'clickData'),
        Input('scatter-options-dropdown', 'value'),
        Input('scatter-options-checklist', 'value')
    )
    def update_scatter(survey_dates, survey_types, depts, clickData, graph_options, hist_options):
        filtered_df = filter_df(df, survey_dates, depts, survey_types)

        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']

        filtered_df['Survey end'] = filtered_df['QUES_PERIOD_END'].astype(str)

        scatter_height = 280

        hist = 'histogram' if 'Scatter histograms' in hist_options else None

        if 'Department' == graph_options:
            fig = px.scatter(filtered_df, x=y, y=x, height=scatter_height,
                        opacity=0.3, trendline="ols", color='GC_ORG_ACRONYM_EN',
                        marginal_x=hist, marginal_y=hist, 
                        )
        elif 'Survey date' == graph_options:
            fig = px.scatter(filtered_df, x=y, y=x, height=scatter_height,
                        opacity=0.3, trendline="ols", color='Survey end',
                        marginal_x=hist, marginal_y=hist, 
                        )
        else:
            fig = px.scatter(filtered_df, x=y, y=x, height=scatter_height,
                        opacity=0.2, trendline="ols", color_discrete_sequence=['ForestGreen'],
                         marginal_x=hist, marginal_y=hist, 
                        )
        fig.update_layout(showlegend=False,
        margin=dict(
            t=30,
            b=0,
            l=0,
            r=10
        ))
        fig.update_layout(plot_bgcolor='white',
        #                   paper_bgcolor=bgcolor, font_color='white'
                        )
        fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)', linecolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)', linecolor='rgba(0,0,0,0.1)')
        fig.update_traces(marker=dict(line=dict(width=1,
                                                color='black')),
                        selector=dict(mode='markers'))

        return fig

    @app.callback(
        Output('animated-scatter', 'children'),
        Input('survey-date-dropdown', 'value'),
        Input('survey-type-dropdown', 'value'),
        Input('department-dropdown', 'value'),
        Input('heatmap', 'clickData'),
        Input('scatter-options-checklist', 'value')
    )
    def update_animated_scatter(survey_dates, survey_types, depts, clickData, graph_options):
        if not 'Scatter animation (slow)' in graph_options:
            return []

        filtered_df = filter_df(df, survey_dates, depts, survey_types)

        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']

        dfx = filtered_df[['GC_ORG_ACRONYM_EN', 'QUES_PERIOD_END', x, y]].dropna()
        print(dfx.head())
        dfx_mean = dfx.groupby(['QUES_PERIOD_END']).mean(numeric_only=True).reset_index()
        print(dfx_mean.head())
        # Add missing rows (where a particular GC_ORG did not give a survey score)
        # Plotly animation requires a complete dataframe (all points present in all frames)
        # <<BEGIN FILLING SECTION>>
        depts = dfx['GC_ORG_ACRONYM_EN'].unique()
        dates = dfx['QUES_PERIOD_END'].unique()
        new_rows = []
        for dept in depts:
            for date in dates:
                if len(dfx[(dfx['GC_ORG_ACRONYM_EN'] == dept) & (dfx['QUES_PERIOD_END'] == date)]) == 0:
                    new_rows.append({
                        'GC_ORG_ACRONYM_EN': dept,
                        'QUES_PERIOD_END': date,
                        x: fill_dept_score(dfx, dept, date, x),
                        y: None
                    })

        dfx = pd.concat([dfx, pd.DataFrame(new_rows)])

        dfx_mean['GC_ORG_ACRONYM_EN'] = 'All Depts Avg'
        dfx = pd.concat([dfx, dfx_mean])
        dfx['Dept Avg'] = 15
        dfx.loc[dfx['GC_ORG_ACRONYM_EN'] == 'All Depts Avg', 'Dept Avg'] = 100

        dfx = dfx.sort_values(['QUES_PERIOD_END'])

        # Forward-fill later missing values (e.g. department skipped a survey)
        dfx[x] = dfx.groupby('GC_ORG_ACRONYM_EN')[x].transform(lambda x: x.ffill())
        dfx[y] = dfx.groupby('GC_ORG_ACRONYM_EN')[y].transform(lambda x: x.ffill())
        
        dfx['Survey End'] = dfx['QUES_PERIOD_END'].astype(str)
        
        # Get the min and max values before backfilling
        miny = min(dfx[y].dropna())
        maxy = max(dfx[y].dropna())

        # Fill early missing values (e.g. department created in 2020, so missing before)
        # dfx[x] = dfx.groupby('GC_ORG_ACRONYM_EN')[x].transform(lambda x: x.backfill())
        # dfx[y] = dfx.groupby('GC_ORG_ACRONYM_EN')[y].transform(lambda x: x.backfill())
        dfx[x] = dfx[x].fillna(-1)
        dfx[y] = dfx[y].fillna(-1)

        # There should be no nulls at this point, but just in case...
        dfx = dfx.dropna()
        
        fig = px.scatter(dfx,
            x=y, y=x,
            symbol_sequence = ['circle', 'triangle-up', 'square', 'x', 'triangle-down', 'cross', 'diamond'],
            animation_frame="Survey End", animation_group="GC_ORG_ACRONYM_EN",
            size='Dept Avg',
            color="GC_ORG_ACRONYM_EN", symbol="GC_ORG_ACRONYM_EN", hover_name="GC_ORG_ACRONYM_EN",
            height=350, opacity=0.6, size_max=20, 
            range_x=[miny - (maxy-miny)*0.03, maxy + (maxy-miny)*0.03],
            range_y=[0.9, 5.1],
            hover_data={'GC_ORG_ACRONYM_EN': False, 
                            'Survey End': True, 
                            x: True,
                            y: True,
                            'Dept Avg': False
                        }
        )

        fig.update_layout(plot_bgcolor='white')
        fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)', linecolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)', linecolor='rgba(0,0,0,0.1)')
        fig.update_traces(marker=dict(line=dict(width=1, color='black')),
                        selector=dict(mode='markers'))

        fig.update_layout(showlegend=False,
        margin=dict(
            t=30,
            b=0,
            l=0,
            r=10
        ))

        try:
            fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
            fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 800
        except:
            fig.update_layout(height=260)
            return [dcc.Graph(figure=fig), html.I('Animation requires more than 1 date.', style={'margin-top': '20px', 'display': 'block'})]

        return dcc.Graph(figure=fig)


    @app.callback(
        Output("filters-collapse", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("filters-collapse", "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open


    @app.callback(
        Output("guide-modal", "is_open"),
        Output("userguide-collapse", "is_open"),
        Output("userguide-readmore-button", "style"),
        [Input("learn-more", "n_clicks"), Input("close", "n_clicks")],
        [Input("userguide-readmore-button", "n_clicks")],
        [State("guide-modal", "is_open")],
    )
    def toggle_modal(n1, n2, n3, is_open):
        if n1 or n2:
            return not is_open, True, {'display': 'none'}
        if n3:
            return True, True, {'display': 'none'}
        return is_open, False, {}


    @app.callback(
        Output('left-panel', 'className'),
        Output('right-panel', 'className'),
        Output('rpanel-close-btn', 'n_clicks'),
        Input("rpanel-close-btn", "n_clicks"),
        Input('heatmap', 'clickData'),
        )
    def toggle_rpanel(collapse_button_clicks, heatmap_clicks):
        if collapse_button_clicks > 0 or heatmap_clicks['points'][0]['z'] == None:
            return '', '', 0
        return 'rpanel-open', 'rpanel-open', 0