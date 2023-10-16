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
import numpy as np


def csfi_corr_plot(corr, title='Correlation with CSFI', maxitems=-1, height=None, width=None):
    if maxitems > 0:
        corr = corr[:maxitems]

    if height is None:
        height = 800
    if width is None:
        width = 1600
    fig = px.imshow(corr, color_continuous_scale='RdBu', width=width,
                    height=height, range_color=[-1, 1], title=title)
    fig.update_xaxes(side="top")
    fig.update_coloraxes(showscale=False)
    return fig


def count_nonzero_points(df):
    score_cols = [c for c in df.columns if c.startswith('S_')]
    metric_cols = [c for c in df.columns if c.startswith('M_')]
    df = df[score_cols + metric_cols]
    numeric_df = df._get_numeric_data()
    mat = numeric_df.to_numpy(dtype=float, na_value=np.nan, copy=False)
    count_arr = np.empty((len(metric_cols), len(score_cols)), dtype=int)
    mask = np.isfinite(mat) & (mat != 0)
    sc_len = len(score_cols)
    for i in range(len(score_cols)): # 0, 1
        for j in range(len(df.columns) - sc_len): # 0, 1, 2
            valid = mask[:, i] & mask[:, j + sc_len]
            c = valid.sum()
            count_arr[j, i] = c
    
    count_df = pd.DataFrame(count_arr, index=metric_cols, columns=score_cols)
    count_df = count_df.loc[count_df.index.sort_values()]
    return count_df

def csfi_corr(df, min_periods=10, method='pearson', mode='overall'):
    metric_cols = [c for c in df.columns if c.startswith('M_')]
    score_cols = [c for c in df.columns if c.startswith('S_')]
    group_cols = [c for c in df.columns if c not in metric_cols + score_cols]
    # print('score cols len', len(score_cols))
    # print(df.columns)
    if mode == 'per_survey':
        # Correlate each survey period separately
        survey_corrs = []
        for d in df['QUES_PERIOD_END'].unique():
            corr = df.query(f'QUES_PERIOD_END == "{d}"').drop(group_cols, axis=1).corr(
                method=method, min_periods=min_periods)[:][score_cols].drop(
                    score_cols)
            survey_corrs.append(corr)
        corr = pd.concat(survey_corrs).groupby(level=0).mean()
    elif mode == 'per_dept':
        # Correlate each department separately
        survey_corrs = []
        for d in df['GC_ORG_ACRONYM_EN'].unique():
            corr = df.query(f'GC_ORG_ACRONYM_EN == "{d}"').drop(group_cols, axis=1).corr(
                method=method, min_periods=min_periods)[:][score_cols].drop(
                    score_cols)
            survey_corrs.append(corr)
        corr = pd.concat(survey_corrs).groupby(level=0).mean()
    else:
        corr = df.drop(group_cols, axis=1).corr(
            method=method, min_periods=min_periods)[score_cols].drop(
                score_cols)
        
    corr = corr.loc[corr.index.sort_values()]
    # print(len(corr.columns), len(corr))
    # counts = count_matrix(df)
    # return corr*counts/counts.max().max()
    return corr


def normalize_metrics(df):
    # Exclude the '_p_ and _avg_' metrics
    norm_cols = [c for c in df.columns if (c[:2] == 'M_') \
                and (c != 'M_Date_Int') \
                and ('_avg' not in c) \
                and ('_median' not in c) \
                and ('_p_' not in c) \
                and ('_percent' not in c.lower()) \
                and ('average' not in c.lower()) \
                and ('_nrml' not in c.lower()) \
                and ('_perEmpl' not in c.lower()) \
                ]
        
    df_nrml = df[norm_cols + ['GC_ORG_NUMBER']].groupby('GC_ORG_NUMBER').apply(lambda x: x/x.mean()).rename(columns={c: c + '_nrml' for c in norm_cols}).drop(columns=['GC_ORG_NUMBER']).reset_index()
    return pd.concat([df_nrml, df], axis=1)


def filter_df(df, survey_dates, depts, survey_types, add_norms=True):
    df = df[(df['QUES_PERIOD_END'].isin(survey_dates))
                     & (df['GC_ORG_ACRONYM_EN'].isin(depts))
                     & (df['SURVEY_TYPE'].isin(survey_types))]
    if add_norms:
        df = normalize_metrics(df)
    return df


def fill_dept_score(df, dept, date, score):
    existing_scores = df[(df['GC_ORG_ACRONYM_EN'] == dept) & (df['QUES_PERIOD_END'] == pd.to_datetime(date)) & ~(df[score].isna())][score]
    if len(existing_scores) > 0:
        return existing_scores.values[0]
    return None


def calc_heatmap_size(corr, maxitems):
    corr = corr.iloc[:maxitems]
    char_size = 10
    square_size = 20
    num_cols = len(corr.columns)
    num_rows = len(corr)
    longest_col = max([len(c) for c in corr.columns])
    longest_row = max([len(c) for c in corr.index])
    # print(f"{num_cols=} {num_rows=} {longest_col=} {longest_row=}")
    width = (char_size * longest_row) + (square_size * num_cols)
    height = (char_size * longest_col) + (square_size * num_rows)
    # print(f'{width=} {height=}')
    return width, height