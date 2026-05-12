import socket
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

from io import BytesIO
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
import webbrowser
from threading import Timer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import plotly.express as px
import plotly.graph_objects as go
import dash_auth
import base64
import os
import atexit
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_USERNAME_PASSWORD_PAIRS = {'Orbit': '123gen'}
WORK_HOURS = 8.5           # productive hours per work day
MIN_TASKS_CLUSTER = 4      # minimum tasks needed before clustering makes sense
MAX_K = 10                 # upper bound for cluster count search

REQUIRED_COLS = ['Timelines', 'Activity_type', 'Task_name', 'Time_taken']
OPTIONAL_COLS = ['Resource', 'Entity', 'Company_code', 'Delivery_Centre', 'ID', 'Cluster']

# ---------------------------------------------------------------------------
# Path / temp-file setup
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

temp_files = []

def cleanup_temp_files():
    for f in temp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

atexit.register(cleanup_temp_files)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
app.title = 'Orbit 2.0'

# Inject global CSS so Dash dropdown internals pick up the dark theme.
# The dcc.Dropdown style prop only styles the wrapper; the inner
# react-select elements need CSS class overrides.
app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        /* ── Dropdown control box ── */
        .Select-control {
            background-color: #001144 !important;
            border-color: rgba(255,255,255,0.3) !important;
        }
        .Select-control:hover {
            border-color: rgba(255,255,255,0.6) !important;
            box-shadow: none !important;
        }
        /* ── Selected value & placeholder text ── */
        .Select-value-label,
        .Select--single > .Select-control .Select-value .Select-value-label {
            color: white !important;
        }
        .Select-placeholder { color: #8899bb !important; }
        .Select-input > input {
            color: white !important;
            background: transparent !important;
        }
        /* ── Dropdown arrow ── */
        .Select-arrow { border-top-color: rgba(255,255,255,0.7) !important; }
        .is-open .Select-arrow {
            border-bottom-color: rgba(255,255,255,0.7) !important;
            border-top-color: transparent !important;
        }
        .Select-clear { color: #aaccff !important; }
        /* ── Open menu ── */
        .Select-menu-outer {
            background-color: #001533 !important;
            border-color: rgba(255,255,255,0.2) !important;
            z-index: 9999 !important;
        }
        .VirtualizedSelectOption {
            color: white !important;
            background-color: #001533 !important;
        }
        .VirtualizedSelectFocusedOption {
            background-color: #003399 !important;
            color: white !important;
        }
        .VirtualizedSelectSelectedOption {
            color: #66aaff !important;
            font-weight: bold;
        }
        /* ── Multi-select tokens ── */
        .Select-multi-value-wrapper .Select-value {
            background-color: #003366 !important;
            border-color: rgba(255,255,255,0.3) !important;
        }
        .Select-multi-value-wrapper .Select-value-label { color: white !important; }
        .Select-multi-value-wrapper .Select-value-icon {
            border-right-color: rgba(255,255,255,0.25) !important;
            color: #aaccff !important;
        }
        /* ── Slider track / handle ── */
        .rc-slider-rail { background-color: rgba(255,255,255,0.2) !important; }
        .rc-slider-track { background-color: #0066cc !important; }
        .rc-slider-handle {
            border-color: #4499ff !important;
            background-color: white !important;
        }
        /* ── Custom scrollbar ── */
        ::-webkit-scrollbar { width: 7px; height: 7px; }
        ::-webkit-scrollbar-track { background: rgba(0,0,20,0.4); }
        ::-webkit-scrollbar-thumb {
            background: rgba(100,150,255,0.4);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(100,150,255,0.7); }
        /* ── Number inputs ── */
        input[type=number] { color: white; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_uploaded_file(contents, filename):
    """Decode base64 upload and return a DataFrame (CSV or Excel)."""
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith('.csv'):
            return pd.read_csv(BytesIO(decoded))
        elif filename.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(BytesIO(decoded))
        return None
    except Exception as e:
        print(f"parse_uploaded_file error: {e}")
        return None


def validate_dataframe(df):
    """Return (is_valid: bool, message: str)."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
    if df['Time_taken'].isna().all():
        return False, "Time_taken column has no valid numeric values."
    return True, "OK"


def find_optimal_k(matrix, max_k=MAX_K):
    """
    Silhouette-score sweep to find the best number of clusters.
    Returns (best_k: int, scores: list[dict]).
    Uses cosine-like behaviour on TF-IDF space by working on the dense form
    only when small; otherwise stays sparse-friendly.
    """
    n = matrix.shape[0]
    effective_max = min(max_k, n - 1)
    if effective_max < 2:
        return max(1, n), []

    scores = []
    for k in range(2, effective_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(matrix)
        try:
            score = silhouette_score(matrix, labels, sample_size=min(1000, n))
        except Exception:
            score = 0.0
        scores.append({'k': k, 'silhouette': round(float(score), 4)})

    best = max(scores, key=lambda x: x['silhouette'])
    return best['k'], scores


def get_cluster_names(vectorizer, km_model, df, activity_col='Activity_type'):
    """
    Build human-readable names for each cluster.
    Format: "<Modal Activity>: <top term1> / <top term2>"
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    df_tmp = df.copy().reset_index(drop=True)
    df_tmp['_lbl'] = km_model.labels_

    label_to_name = {}
    for k in range(km_model.n_clusters):
        centroid = km_model.cluster_centers_[k]
        top_idx = centroid.argsort()[-3:][::-1]
        terms = [feature_names[j] for j in top_idx if feature_names[j].strip()]

        subset = df_tmp[df_tmp['_lbl'] == k]
        if not subset.empty and activity_col in subset.columns:
            modal_act = subset[activity_col].mode().iloc[0]
        else:
            modal_act = f'Group {k + 1}'

        term_str = ' / '.join(terms[:2]) if terms else ''
        label_to_name[k] = f"{modal_act}: {term_str}" if term_str else modal_act

    return [label_to_name[l] for l in km_model.labels_], label_to_name


def perform_clustering(df, quantile_value=75, use_time_feature=True):
    """
    Cluster tasks by TF-IDF similarity (optionally combined with scaled
    Time_taken), auto-select K via silhouette score, then apply quantile-
    based level loading.

    Returns (enriched_df, label_name_map, k_scores).
    """
    if df is None or len(df) < MIN_TASKS_CLUSTER:
        return df, {}, []

    tasks = df['Task_name'].fillna('').astype(str).tolist()

    try:
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', min_df=1)
        tfidf = vectorizer.fit_transform(tasks)

        if use_time_feature and 'Time_taken' in df.columns:
            scaler = MinMaxScaler()
            time_arr = df['Time_taken'].fillna(0).values.reshape(-1, 1)
            time_scaled = scaler.fit_transform(time_arr)
            combined = hstack([tfidf, csr_matrix(time_scaled * 0.3)])  # weight 0.3 so text dominates
        else:
            combined = tfidf

        best_k, k_scores = find_optimal_k(combined)
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
        km.fit(combined)

        cluster_labels, label_name_map = get_cluster_names(vectorizer, km, df.reset_index(drop=True))

        result = df.copy().reset_index(drop=True)
        result['Task_cluster'] = cluster_labels
        result['Cluster_id'] = km.labels_

        # --- Level loading ---
        q = quantile_value / 100.0
        targets = result.groupby('Task_cluster')['Time_taken'].quantile(q).reset_index()
        targets.rename(columns={'Time_taken': 'Standard_time'}, inplace=True)
        result = pd.merge(result, targets, on='Task_cluster', how='left')

        result['Level_loaded_time'] = result.apply(
            lambda r: r['Standard_time'] if r['Time_taken'] > r['Standard_time'] else r['Time_taken'],
            axis=1
        )
        result['Level_Loaded_Flag'] = (result['Level_loaded_time'] != result['Time_taken']).map({True: 'Yes', False: 'No'})
        result['Time_saved'] = result['Time_taken'] - result['Level_loaded_time']
        result['Time_taken_FTE'] = result['Time_taken'] / (60 * WORK_HOURS)
        result['Level_loaded_FTE'] = result['Level_loaded_time'] / (60 * WORK_HOURS)

        # --- Outlier flag: > 2 std devs above cluster mean ---
        stats = result.groupby('Task_cluster')['Time_taken'].agg(['mean', 'std']).reset_index()
        stats.columns = ['Task_cluster', 'Cluster_mean', 'Cluster_std']
        result = pd.merge(result, stats, on='Task_cluster', how='left')
        result['Cluster_std'] = result['Cluster_std'].fillna(0)
        result['Is_Outlier'] = result['Time_taken'] > (result['Cluster_mean'] + 2 * result['Cluster_std'])

        return result, label_name_map, k_scores

    except Exception as e:
        print(f"perform_clustering error: {e}")
        return df, {}, []


def make_dark_chart_layout(title='', extra=None):
    """Return a standard dark-themed Plotly layout dict."""
    layout = dict(
        plot_bgcolor='rgba(0,0,30,0.6)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Calibri', size=12),
        title=dict(text=title, font=dict(color='white', family='Calibri', size=16), x=0.5),
        xaxis=dict(gridcolor='rgba(255,255,255,0.12)', tickfont=dict(color='white')),
        yaxis=dict(gridcolor='rgba(255,255,255,0.12)', tickfont=dict(color='white')),
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.3)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1),
        margin=dict(l=50, r=50, t=70, b=50),
    )
    if extra:
        layout.update(extra)
    return layout


# ---------------------------------------------------------------------------
# Shared styles
# ---------------------------------------------------------------------------
PAGE_STYLE = {
    'background': 'linear-gradient(135deg, #000022 0%, #000055 50%, #000088 100%)',
    'minHeight': '100vh',
    'color': 'white',
    'fontFamily': 'Calibri, Roboto, sans-serif',
    'padding': '20px',
}

CARD_STYLE = {
    'backgroundColor': 'rgba(255,255,255,0.08)',
    'border': '1px solid rgba(255,255,255,0.15)',
    'borderRadius': '8px',
    'padding': '16px',
    'marginBottom': '12px',
}

TAB_STYLE = {
    'backgroundColor': 'rgba(0,0,60,0.7)',
    'color': '#aaa',
    'border': '1px solid rgba(255,255,255,0.1)',
    'borderBottom': 'none',
    'padding': '8px 16px',
    'fontFamily': 'Calibri',
}

TAB_SELECTED_STYLE = {
    'backgroundColor': 'rgba(0,80,180,0.8)',
    'color': 'white',
    'border': '1px solid rgba(255,255,255,0.3)',
    'borderBottom': 'none',
    'padding': '8px 16px',
    'fontFamily': 'Calibri',
    'fontWeight': 'bold',
}

BUTTON_STYLE = {
    'backgroundColor': '#0066cc',
    'color': 'white',
    'border': 'none',
    'padding': '8px 18px',
    'cursor': 'pointer',
    'borderRadius': '4px',
    'marginRight': '8px',
    'fontFamily': 'Calibri',
    'fontSize': '14px',
}

DROPDOWN_STYLE = {
    'backgroundColor': '#001144',
    'color': 'white',
    'border': '1px solid rgba(255,255,255,0.3)',
    'borderRadius': '4px',
}

LABEL_STYLE = {'fontFamily': 'Calibri', 'fontSize': '14px', 'marginTop': '10px', 'marginBottom': '4px', 'color': '#ccc'}

# ---------------------------------------------------------------------------
# Action table initial data
# ---------------------------------------------------------------------------
_action_cols = ['Insight', 'Action', 'Owner', 'Target Date']
action_table_init = pd.DataFrame('', index=range(6), columns=_action_cols).to_dict('records')

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(style=PAGE_STYLE, children=[

    # ---- Header ----
    html.Div(style={'textAlign': 'center', 'marginBottom': '20px'}, children=[
        html.H1('ORBIT 2.0', style={'fontSize': '36px', 'fontWeight': 'bold', 'color': '#66aaff', 'margin': '0'}),
        html.P('Peak Load Analysis & Workload Standardisation Tool',
               style={'fontSize': '16px', 'color': '#aabbdd', 'marginTop': '4px'}),
    ]),

    # ---- Upload ----
    html.Div(style={**CARD_STYLE, 'display': 'flex', 'alignItems': 'center', 'gap': '20px'}, children=[
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV / Excel File', style=BUTTON_STYLE),
            multiple=False
        ),
        html.Div(id='file-upload-status', style={'fontSize': '15px', 'color': 'lightgreen'}),
        html.Button('Download Input Template', id='input-sheet', n_clicks=0,
                    style={**BUTTON_STYLE, 'backgroundColor': '#444', 'marginLeft': 'auto'}),
        dcc.Download(id='input-sheet-download'),
    ]),

    # ---- Scorecard (hidden until data loaded) ----
    html.Div(id='scorecard', style={'display': 'none'}, children=[
        html.Div(style={'display': 'flex', 'gap': '12px', 'marginBottom': '16px', 'flexWrap': 'wrap'}, children=[
            html.Div(id='sc-peak-fte',   style={**CARD_STYLE, 'flex': '1', 'textAlign': 'center', 'minWidth': '140px'}),
            html.Div(id='sc-peak-day',   style={**CARD_STYLE, 'flex': '1', 'textAlign': 'center', 'minWidth': '140px'}),
            html.Div(id='sc-total-fte',  style={**CARD_STYLE, 'flex': '1', 'textAlign': 'center', 'minWidth': '140px'}),
            html.Div(id='sc-tasks',      style={**CARD_STYLE, 'flex': '1', 'textAlign': 'center', 'minWidth': '140px'}),
            html.Div(id='sc-resources',  style={**CARD_STYLE, 'flex': '1', 'textAlign': 'center', 'minWidth': '140px'}),
            html.Div(id='sc-top-act',    style={**CARD_STYLE, 'flex': '1', 'textAlign': 'center', 'minWidth': '140px'}),
        ])
    ]),

    # ---- Stores ----
    dcc.Store(id='uploaded-file-path', storage_type='memory'),
    dcc.Store(id='prediction-data-store', storage_type='memory'),

    # ---- Main Tabs ----
    dcc.Tabs(style={'marginBottom': '0'}, children=[

        # ==============================
        # TAB 0 – How to Use
        # ==============================
        dcc.Tab(label='How to Use', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={**CARD_STYLE, 'maxWidth': '900px', 'margin': '20px auto'}, children=[
                html.H2('What is Orbit?', style={'color': '#66aaff'}),
                html.P(
                    'Orbit is a Peak Load Analysis tool built for operations and planning teams. '
                    'It helps you understand when your team is overloaded, which tasks are driving that load, '
                    'and how much time could be saved by standardising similar tasks.',
                    style={'lineHeight': '1.7'}
                ),

                html.H3('When to Use Orbit', style={'color': '#66aaff', 'marginTop': '24px'}),
                html.Ul([
                    html.Li('You suspect certain days of the month are overloaded compared to others.'),
                    html.Li('You want to identify which activity types or resources are driving peak load.'),
                    html.Li('You want to quantify time-saving potential from process standardisation.'),
                    html.Li('You are preparing a capacity planning or workload rebalancing exercise.'),
                    html.Li('You want evidence-backed targets for task completion time ("time boxing").'),
                ], style={'lineHeight': '2.0'}),

                html.H3('Required Data Format', style={'color': '#66aaff', 'marginTop': '24px'}),
                html.P('Upload a CSV or Excel file with at least these columns (names must match exactly):'),
                dash_table.DataTable(
                    columns=[{'name': c, 'id': c} for c in ['Column', 'Type', 'Description', 'Example']],
                    data=[
                        {'Column': 'Timelines',     'Type': 'Text',   'Description': 'Work day or period label',          'Example': 'Day 1, Week 3, 01-Jan'},
                        {'Column': 'Activity_type', 'Type': 'Text',   'Description': 'Category of work',                  'Example': 'Reconciliation, Reporting'},
                        {'Column': 'Task_name',     'Type': 'Text',   'Description': 'Specific task description',         'Example': 'Bank rec – Entity ABC'},
                        {'Column': 'Time_taken',    'Type': 'Number', 'Description': 'Minutes spent on the task',         'Example': '45, 120, 30'},
                        {'Column': 'Resource',      'Type': 'Text',   'Description': '(Optional) Person or team',         'Example': 'Alice, Team A'},
                        {'Column': 'Entity',        'Type': 'Text',   'Description': '(Optional) Business entity',        'Example': 'Corp, Division X'},
                    ],
                    style_table={'backgroundColor': 'rgba(0,0,60,0.5)'},
                    style_header={'backgroundColor': '#003366', 'color': 'white', 'fontWeight': 'bold'},
                    style_cell={'backgroundColor': 'rgba(0,0,40,0.8)', 'color': 'white', 'textAlign': 'left',
                                'padding': '8px', 'fontFamily': 'Calibri'},
                ),

                html.H3('How to Read the Scorecard', style={'color': '#66aaff', 'marginTop': '24px'}),
                html.Ul([
                    html.Li([html.B('Peak Load (FTE): '), 'The total task time on the busiest day, expressed as Full-Time Equivalents (FTE = total minutes ÷ 510 mins per day). e.g. 3.5 FTE means the peak day requires 3.5 people working a full day.']),
                    html.Li([html.B('Total FTE: '), 'Sum of all task time across all days expressed as FTE.']),
                ], style={'lineHeight': '2.0'}),

                html.H3('How Peak Load Clustering Works', style={'color': '#66aaff', 'marginTop': '24px'}),
                html.P('The clustering tab groups similar tasks together using their names and time taken, then calculates a "standard time" for each group:', style={'lineHeight': '1.7'}),
                html.Ol([
                    html.Li('Tasks are converted to numeric vectors using TF-IDF on their names, combined with a scaled version of their time taken.'),
                    html.Li('The algorithm automatically tests cluster counts from 2 to 10 and picks the number (K) with the highest Silhouette Score — a measure of how well-separated the clusters are.'),
                    html.Li('Each cluster gets a "Standard Time" equal to the Nth percentile of task times within that cluster. The percentile is controlled by the Standardisation slider (75 = aggressive, 90 = lenient).'),
                    html.Li('Tasks that took LONGER than the standard time for their cluster are "level loaded" — their time is reduced to the standard. The difference is your Time Saved.'),
                    html.Li('Tasks more than 2 standard deviations above their cluster mean are flagged as outliers — priority candidates for process improvement or coaching.'),
                ], style={'lineHeight': '2.2'}),

                html.H3('Interpreting the Standardisation Slider', style={'color': '#66aaff', 'marginTop': '24px'}),
                html.Ul([
                    html.Li([html.B('Aggressive (75th percentile): '), '75% of tasks in each cluster would be at or below the standard time. More tasks get level-loaded, larger savings estimate — use when you believe significant improvement is achievable.']),
                    html.Li([html.B('Lenient (90th percentile): '), 'Only the top 10% of task times get level-loaded. Conservative estimate — use for initial baselining or when process maturity is high.']),
                ], style={'lineHeight': '2.0'}),

                html.H3('Workflow', style={'color': '#66aaff', 'marginTop': '24px'}),
                html.Ol([
                    html.Li('Upload your task data CSV/Excel.'),
                    html.Li('Check the Scorecard — note your peak day and peak load FTE.'),
                    html.Li('Open "Workload Analysis" → Stacked Chart to see load distribution across days.'),
                    html.Li('Set threshold lines to represent your team capacity (e.g. 10 FTE).'),
                    html.Li('Open "Peak Load Clustering" — select the peak work day and the activity type you want to analyse.'),
                    html.Li('Review the cluster chart and level-load comparison. Export recommendations.'),
                    html.Li('Fill the Action Planning table and export it as your action log.'),
                ], style={'lineHeight': '2.2'}),
            ])
        ]),

        # ==============================
        # TAB 1 – Workload Analysis
        # ==============================
        dcc.Tab(label='Workload Analysis', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            dcc.Tabs(style={'marginTop': '10px'}, children=[

                # -- Bar Chart --
                dcc.Tab(label='Bar Chart', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                    html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[
                        html.H3(id='fte', style={'color': '#ffcc44', 'margin': '0 0 12px 0'}),
                        html.Div([
                            html.Label('Group by:', style=LABEL_STYLE),
                            dcc.Dropdown(
                                id='x-axis-column',
                                options=[
                                    {'label': 'Activity Type', 'value': 'Activity_type'},
                                    {'label': 'Timelines (Work Day)', 'value': 'Timelines'},
                                    {'label': 'Resource', 'value': 'Resource'},
                                    {'label': 'Delivery Centre', 'value': 'Delivery_Centre'},
                                    {'label': 'Cluster / Team', 'value': 'Cluster'},
                                ],
                                value='Activity_type',
                                style=DROPDOWN_STYLE,
                            ),
                        ], style={'width': '30%'}),
                        dcc.Graph(id='bar-chart'),
                        html.Div(id='bar-chart-summary', style={'fontSize': '13px', 'color': '#aaccff', 'marginTop': '6px'}),
                    ])
                ]),

                # -- Box Plot --
                dcc.Tab(label='Box Plot', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                    html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[
                        html.Div([
                            html.Label('Group by:', style=LABEL_STYLE),
                            dcc.Dropdown(
                                id='multi-x-axis-columns',
                                options=[
                                    {'label': 'Entity', 'value': 'Entity'},
                                    {'label': 'Resource', 'value': 'Resource'},
                                    {'label': 'Activity Type', 'value': 'Activity_type'},
                                    {'label': 'Timelines', 'value': 'Timelines'},
                                ],
                                value='Resource',
                                style=DROPDOWN_STYLE,
                            ),
                        ], style={'width': '30%'}),
                        dcc.Graph(id='box-plot'),
                        html.Div(id='box-plot-summary', style={'fontSize': '13px', 'color': '#aaccff', 'marginTop': '6px'}),
                    ])
                ]),

                # -- Bubble Chart --
                dcc.Tab(label='Bubble Chart', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                    html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[
                        html.P('Bubble size = time taken. X = Work Day, Y = Resource, Colour = Activity Type.',
                               style={'color': '#aabbdd', 'fontSize': '13px'}),
                        dcc.Graph(id='bubble-plot'),
                        html.Div(id='bubble-plot-summary', style={'fontSize': '13px', 'color': '#aaccff', 'marginTop': '6px'}),
                    ])
                ]),

                # -- Heatmap --
                dcc.Tab(label='Heatmap', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                    html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[
                        html.Div([
                            html.Label('X-Axis:', style=LABEL_STYLE),
                            dcc.Dropdown(id='x-axis-dropdown',
                                         options=[
                                             {'label': 'Timelines', 'value': 'Timelines'},
                                             {'label': 'Activity Type', 'value': 'Activity_type'},
                                             {'label': 'Resource', 'value': 'Resource'},
                                             {'label': 'Entity', 'value': 'Entity'},
                                         ],
                                         value='Timelines', style=DROPDOWN_STYLE),
                            html.Label('Y-Axis:', style=LABEL_STYLE),
                            dcc.Dropdown(id='y-axis-dropdown',
                                         options=[
                                             {'label': 'Activity Type', 'value': 'Activity_type'},
                                             {'label': 'Timelines', 'value': 'Timelines'},
                                             {'label': 'Resource', 'value': 'Resource'},
                                             {'label': 'Entity', 'value': 'Entity'},
                                         ],
                                         value='Activity_type', style=DROPDOWN_STYLE),
                        ], style={'width': '30%'}),
                        dcc.Graph(id='heatmap'),
                        html.Div(id='heatmap-summary', style={'fontSize': '13px', 'color': '#aaccff', 'marginTop': '6px'}),
                    ])
                ]),

                # -- Stacked Workload Chart --
                dcc.Tab(label='Stacked Workload', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                    html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[
                        html.Div([
                            html.Label('Capacity Line 1 (Red) — FTE:', style=LABEL_STYLE),
                            dcc.Input(id='threshold-line1', type='number', value=0, step=0.5,
                                      style={'width': '100%', 'marginBottom': '8px', 'backgroundColor': '#001', 'color': 'white', 'border': '1px solid #555', 'padding': '4px'}),
                            html.Label('Capacity Line 2 (Orange) — FTE:', style=LABEL_STYLE),
                            dcc.Input(id='threshold-line2', type='number', value=0, step=0.5,
                                      style={'width': '100%', 'marginBottom': '8px', 'backgroundColor': '#001', 'color': 'white', 'border': '1px solid #555', 'padding': '4px'}),
                            html.P('Set these to your team\'s capacity in FTE to see which days breach it.',
                                   style={'color': '#aabbdd', 'fontSize': '12px'}),
                        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),
                        html.Div([
                            dcc.Graph(id='stacked-bar-chart'),
                        ], style={'width': '78%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                        html.Div(style={'marginTop': '20px'}, children=[
                            html.H4('Top 3 Activities by FTE per Work Day', style={'color': '#66aaff'}),
                            dash_table.DataTable(
                                id='top-activities-table',
                                columns=[
                                    {'name': 'Work Day', 'id': 'Timelines'},
                                    {'name': 'Activity Type', 'id': 'Activity_type'},
                                    {'name': 'FTE', 'id': 'Time_taken_FTE'},
                                ],
                                data=[],
                                style_table={'overflowX': 'auto'},
                                style_header={'backgroundColor': '#003366', 'color': 'white', 'fontWeight': 'bold'},
                                style_cell={'textAlign': 'center', 'backgroundColor': 'rgba(0,0,40,0.8)',
                                            'color': 'white', 'fontFamily': 'Calibri', 'padding': '8px'},
                                page_size=12,
                            ),
                        ]),
                    ])
                ]),

            ])
        ]),

        # ==============================
        # TAB 1b – Resource Utilization
        # ==============================
        dcc.Tab(label='Resource Utilization', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[
                html.H3('Who Is Doing What — and When?', style={'color': '#66aaff'}),
                html.P(
                    'Understand how work is distributed across your team. '
                    'Resources with >1.0 FTE on a single day are flagged as overloaded.',
                    style={'color': '#aabbdd', 'fontSize': '14px'}
                ),
                html.Div(id='resource-overload-alert',
                         style={'color': '#ff7755', 'fontSize': '14px', 'fontWeight': 'bold',
                                'marginBottom': '10px'}),

                # Resource × Workday heatmap
                dcc.Graph(id='resource-heatmap'),

                # Delivery Centre / Entity split (if available)
                dcc.Graph(id='dc-entity-bar'),

                html.Hr(style={'borderColor': 'rgba(255,255,255,0.15)', 'margin': '20px 0'}),
                html.H4('Task Duration Comparison Across Resources', style={'color': '#66aaff'}),
                html.P('Select an activity type to see whether certain resources consistently '
                       'take longer than others on the same work — a coaching/process signal.',
                       style={'color': '#aabbdd', 'fontSize': '13px'}),
                html.Div([
                    html.Label('Activity Type:', style=LABEL_STYLE),
                    dcc.Dropdown(id='resource-activity-selector', style=DROPDOWN_STYLE,
                                 placeholder='Select an activity type…'),
                ], style={'width': '40%', 'marginBottom': '12px'}),
                dcc.Graph(id='resource-task-box'),
            ])
        ]),

        # ==============================
        # TAB 2 – Peak Load Clustering
        # ==============================
        dcc.Tab(label='Peak Load Clustering', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[

                html.H3('Identify & Standardise Similar Tasks', style={'color': '#66aaff'}),
                html.P(
                    'Select a peak work day and activity type. Orbit will automatically find the best '
                    'number of task groups (clusters), set a standard time for each, and quantify the '
                    'time that could be saved through standardisation.',
                    style={'color': '#aabbdd', 'fontSize': '14px', 'lineHeight': '1.7'}
                ),

                # Controls row
                html.Div(style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap', 'marginBottom': '16px'}, children=[
                    html.Div(style={'flex': '1', 'minWidth': '180px'}, children=[
                        html.Label('Peak Work Day:', style=LABEL_STYLE),
                        dcc.Dropdown(id='Peak-workday', options=[], style=DROPDOWN_STYLE),
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '180px'}, children=[
                        html.Label('Activity Type:', style=LABEL_STYLE),
                        dcc.Dropdown(id='Activity-type', options=[], style=DROPDOWN_STYLE),
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '200px'}, children=[
                        html.Label('Standardisation Aggressiveness:', style=LABEL_STYLE),
                        dcc.Slider(
                            id='percentile-slider', min=75, max=90, value=80, step=5,
                            marks={
                                75: {'label': '75 – Aggressive', 'style': {'color': '#fff', 'fontSize': '11px'}},
                                80: {'label': '80', 'style': {'color': '#fff', 'fontSize': '11px'}},
                                85: {'label': '85', 'style': {'color': '#fff', 'fontSize': '11px'}},
                                90: {'label': '90 – Lenient', 'style': {'color': '#fff', 'fontSize': '11px'}},
                            },
                            tooltip={'placement': 'bottom', 'always_visible': False}
                        ),
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '140px', 'display': 'flex', 'alignItems': 'center'}, children=[
                        dcc.Checklist(
                            id='use-time-feature',
                            options=[{'label': ' Include task time in clustering', 'value': 'yes'}],
                            value=['yes'],
                            style={'color': '#ccc', 'fontSize': '13px'},
                        )
                    ]),
                ]),

                # Cluster quality metrics bar
                html.Div(id='cluster-metrics', style={
                    'backgroundColor': 'rgba(0,60,120,0.4)',
                    'border': '1px solid rgba(100,150,255,0.3)',
                    'borderRadius': '6px',
                    'padding': '12px 20px',
                    'marginBottom': '16px',
                    'fontSize': '14px',
                }),

                # Level load headline
                html.H3(id='level-fte', style={'color': '#44ff88', 'textAlign': 'center'}),

                # Charts
                dcc.Graph(id='cluster-chart'),
                dcc.Graph(id='level-load-chart'),
                dcc.Graph(id='savings-by-cluster-chart'),

                # Silhouette chart (collapsible-ish)
                html.Details([
                    html.Summary('Show Optimal K Selection (Silhouette Scores)',
                                 style={'cursor': 'pointer', 'color': '#aabbdd', 'marginBottom': '8px'}),
                    dcc.Graph(id='silhouette-chart'),
                ], style={'marginTop': '10px'}),

                # Export buttons
                html.Div(style={'marginTop': '16px', 'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}, children=[
                    html.Button('Export Level Loading Recommendations', id='export-button-pred', n_clicks=0, style=BUTTON_STYLE),
                    html.Button('Export Top 3 Workdays & Activities', id='export-multi-button', n_clicks=0, style=BUTTON_STYLE),
                    dcc.Download(id='prediction-data-download'),
                    dcc.Download(id='multi-data-download'),
                ]),
            ]),
        ]),

        # ==============================
        # TAB 3 – What-If Load Simulator
        # ==============================
        dcc.Tab(label='What-If Simulator', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[
                html.H3('What If We Moved Work Away from the Peak Day?', style={'color': '#66aaff'}),
                html.P(
                    'Drag a percentage of an activity type from your peak day to one or more other days. '
                    'Orbit recalculates the full load curve instantly so you can see whether rescheduling '
                    'is worth it — and which specific tasks would move.',
                    style={'color': '#aabbdd', 'fontSize': '14px', 'lineHeight': '1.7'}
                ),

                # ── Controls ──
                html.Div(style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap',
                                'marginBottom': '20px', 'alignItems': 'flex-end'}, children=[
                    html.Div(style={'flex': '1', 'minWidth': '160px'}, children=[
                        html.Label('Move FROM (source day):', style=LABEL_STYLE),
                        dcc.Dropdown(id='whatif-source-day', options=[], style=DROPDOWN_STYLE,
                                     placeholder='Select source day…'),
                    ]),
                    html.Div(style={'flex': '2', 'minWidth': '200px'}, children=[
                        html.Label('Move TO (target day/s):', style=LABEL_STYLE),
                        dcc.Dropdown(id='whatif-target-days', options=[], multi=True,
                                     style=DROPDOWN_STYLE, placeholder='Select one or more days…'),
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '160px'}, children=[
                        html.Label('Activity Type (blank = all):', style=LABEL_STYLE),
                        dcc.Dropdown(id='whatif-activity', options=[], style=DROPDOWN_STYLE,
                                     placeholder='All activities'),
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '200px'}, children=[
                        html.Label(id='whatif-slider-label',
                                   children='Shift 20% of selected work',
                                   style={**LABEL_STYLE, 'color': '#ffdd88'}),
                        dcc.Slider(id='whatif-pct', min=5, max=100, value=20, step=5,
                                   marks={
                                       5:   {'label': '5%',   'style': {'color': '#ccc', 'fontSize': '11px'}},
                                       25:  {'label': '25%',  'style': {'color': '#ccc', 'fontSize': '11px'}},
                                       50:  {'label': '50%',  'style': {'color': '#ccc', 'fontSize': '11px'}},
                                       75:  {'label': '75%',  'style': {'color': '#ccc', 'fontSize': '11px'}},
                                       100: {'label': '100%', 'style': {'color': '#ccc', 'fontSize': '11px'}},
                                   }),
                    ]),
                ]),

                # ── Headline ──
                html.Div(id='whatif-headline', style={
                    'backgroundColor': 'rgba(0,80,40,0.4)',
                    'border': '1px solid rgba(100,255,150,0.3)',
                    'borderRadius': '6px',
                    'padding': '12px 20px',
                    'marginBottom': '16px',
                    'fontSize': '15px',
                    'color': '#88ffaa',
                }),

                # ── Before / After chart ──
                dcc.Graph(id='whatif-chart'),

                # ── Tasks that would move ──
                html.Details(style={'marginTop': '16px'}, children=[
                    html.Summary('Show tasks that would be rescheduled',
                                 style={'cursor': 'pointer', 'color': '#aabbdd',
                                        'marginBottom': '8px', 'fontSize': '14px'}),
                    dash_table.DataTable(
                        id='whatif-task-table',
                        columns=[{'name': c, 'id': c} for c in
                                 ['Task_name', 'Activity_type', 'Resource',
                                  'Time_taken', 'Time_taken_hrs', 'FTE_contribution']],
                        data=[],
                        page_size=15,
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#003366', 'color': 'white',
                                      'fontWeight': 'bold'},
                        style_cell={'backgroundColor': 'rgba(0,0,40,0.8)', 'color': 'white',
                                    'fontFamily': 'Calibri', 'padding': '8px',
                                    'textAlign': 'left'},
                    ),
                ]),
            ])
        ]),

        # ==============================
        # TAB 4 – Action Planning
        # ==============================
        dcc.Tab(label='Action Planning', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={**CARD_STYLE, 'marginTop': '12px'}, children=[
                html.H3('Document Actions from Your Analysis', style={'color': '#66aaff'}),
                html.P('Record insights and action owners here. Export as Excel when done.',
                       style={'color': '#aabbdd', 'fontSize': '14px'}),
                dash_table.DataTable(
                    id='data-table',
                    columns=[{'name': c, 'id': c, 'editable': True} for c in _action_cols],
                    data=action_table_init,
                    editable=True,
                    row_deletable=True,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#003366', 'color': 'white', 'textAlign': 'center', 'fontWeight': 'bold'},
                    style_cell={
                        'textAlign': 'center',
                        'backgroundColor': 'rgba(0,0,40,0.8)',
                        'color': 'white',
                        'border': '1px solid rgba(255,255,255,0.15)',
                        'fontFamily': 'Calibri',
                        'padding': '8px',
                    },
                ),
                html.Div(style={'marginTop': '12px'}, children=[
                    html.Button('Export to Excel', id='export-button', n_clicks=0, style=BUTTON_STYLE),
                    dcc.Download(id='data-download'),
                ]),
            ]),
        ]),

    ]),  # end main Tabs

    # ---- Exit ----
    html.Div(style={'textAlign': 'center', 'marginTop': '30px'}, children=[
        html.Button('Exit Application', id='exit-button', n_clicks=0,
                    style={**BUTTON_STYLE, 'backgroundColor': '#cc2200', 'padding': '10px 28px'}),
    ]),
    html.Div(id='dummy-output', style={'display': 'none'}),
])


# ===========================================================================
# Callbacks
# ===========================================================================

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
@app.callback(
    [Output('uploaded-file-path', 'data'),
     Output('file-upload-status', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def handle_file_upload(contents, filename):
    if contents is None:
        return None, 'No file uploaded yet.'
    try:
        df = parse_uploaded_file(contents, filename)
        if df is None:
            return None, 'Error: Only CSV and Excel (.xlsx/.xls) files are supported.'
        ok, msg = validate_dataframe(df)
        if not ok:
            return None, f'Validation error: {msg}'
        # Save temp file for callbacks that re-read it
        temp_dir = os.path.join(base_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f'temp_{filename}')
        # Save as CSV so all callbacks can read uniformly
        df.to_csv(temp_path + '.csv', index=False)
        actual_path = temp_path + '.csv'
        temp_files.append(actual_path)
        return actual_path, f"✓ '{filename}' uploaded — {len(df):,} rows, {df['Timelines'].nunique()} work days."
    except Exception as e:
        print(f'handle_file_upload error: {e}')
        return None, 'Error: Unable to process the uploaded file.'


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
@app.callback(
    [Output('scorecard', 'style'),
     Output('sc-peak-fte', 'children'),
     Output('sc-peak-day', 'children'),
     Output('sc-total-fte', 'children'),
     Output('sc-tasks', 'children'),
     Output('sc-resources', 'children'),
     Output('sc-top-act', 'children')],
    [Input('uploaded-file-path', 'data')]
)
def update_scorecard(file_path):
    hidden = {'display': 'none'}
    visible = {'display': 'block'}
    if not file_path:
        return hidden, '', '', '', '', '', ''
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        df['FTE'] = df['Time_taken'] / (60 * WORK_HOURS)

        daily = df.groupby('Timelines')['FTE'].sum()
        peak_fte = daily.max()
        peak_day = daily.idxmax()
        total_fte = df['FTE'].sum()
        n_tasks = len(df)
        n_resources = df['Resource'].nunique() if 'Resource' in df.columns else 'N/A'
        top_act = df.groupby('Activity_type')['FTE'].sum().idxmax() if 'Activity_type' in df.columns else 'N/A'

        def card(label, value):
            return [
                html.Div(label, style={'fontSize': '11px', 'color': '#99aacc', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                html.Div(str(value), style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#66aaff', 'marginTop': '4px'}),
            ]

        return (
            visible,
            card('Peak Load', f'{peak_fte:.1f} FTE'),
            card('Peak Work Day', peak_day),
            card('Total FTE (All Days)', f'{total_fte:.1f}'),
            card('Total Tasks', f'{n_tasks:,}'),
            card('Resources', str(n_resources)),
            card('Top Activity', top_act),
        )
    except Exception as e:
        print(f'update_scorecard error: {e}')
        return hidden, '', '', '', '', '', ''


# ---------------------------------------------------------------------------
# Dropdowns for clustering tab
# ---------------------------------------------------------------------------
@app.callback(
    [Output('Peak-workday', 'options'),
     Output('Activity-type', 'options')],
    [Input('uploaded-file-path', 'data')]
)
def update_cluster_dropdowns(file_path):
    if not file_path:
        return [], []
    try:
        df = pd.read_csv(file_path)
        days = sorted(df['Timelines'].dropna().unique().tolist())
        acts = sorted(df['Activity_type'].dropna().unique().tolist())
        return ([{'label': d, 'value': d} for d in days],
                [{'label': a, 'value': a} for a in acts])
    except Exception as e:
        print(f'update_cluster_dropdowns error: {e}')
        return [], []


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
@app.callback(
    [Output('fte', 'children'),
     Output('bar-chart', 'figure'),
     Output('bar-chart-summary', 'children')],
    [Input('x-axis-column', 'value'),
     Input('uploaded-file-path', 'data')]
)
def update_bar_chart(x_col, file_path):
    if not file_path:
        return '', {}, ''
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        if x_col not in df.columns:
            return f'Column "{x_col}" not in data.', {}, ''
        df['FTE'] = df['Time_taken'] / (60 * WORK_HOURS)

        daily = df.groupby('Timelines')['FTE'].sum()
        peak_fte = daily.max()
        peak_day = daily.idxmax()

        grouped = df.groupby(x_col)['FTE'].sum().reset_index().sort_values('FTE', ascending=False)
        fig = px.bar(grouped, x=x_col, y='FTE',
                     color='FTE', color_continuous_scale='blues',
                     text='FTE',
                     title=f'Total FTE by {x_col}')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(**make_dark_chart_layout())
        fig.update_coloraxes(showscale=False)

        summary = (f"Peak single-day load: {peak_fte:.2f} FTE (on {peak_day}). "
                   f"Max {x_col}: {grouped.iloc[0][x_col]} ({grouped.iloc[0]['FTE']:.2f} FTE). "
                   f"Average across groups: {grouped['FTE'].mean():.2f} FTE.")
        return f'Peak Load = {peak_fte:.2f} FTE  |  Peak Day = {peak_day}', fig, summary
    except Exception as e:
        print(f'update_bar_chart error: {e}')
        return '', {}, ''


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------
@app.callback(
    [Output('box-plot', 'figure'),
     Output('box-plot-summary', 'children')],
    [Input('multi-x-axis-columns', 'value'),
     Input('uploaded-file-path', 'data')]
)
def update_box_plot(x_col, file_path):
    if not file_path or not x_col:
        return {}, ''
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        if x_col not in df.columns:
            return {}, f'Column "{x_col}" not in data.'
        df['Time_taken_hours'] = df['Time_taken'] / 60

        hover = [c for c in ['Task_name', 'ID', 'Timelines'] if c in df.columns]
        fig = px.box(df, x=x_col, y='Time_taken_hours',
                     title=f'Task Time Distribution by {x_col}',
                     points='all', hover_data=hover)
        fig.update_traces(marker_color='#4499ff', marker=dict(outliercolor='#ff4444'))
        fig.update_layout(**make_dark_chart_layout())

        q1 = df['Time_taken_hours'].quantile(0.25)
        q3 = df['Time_taken_hours'].quantile(0.75)
        iqr = q3 - q1
        outliers = df[df['Time_taken_hours'] > q3 + 1.5 * iqr]
        summary = (f"IQR: {iqr:.2f} h (Q1={q1:.2f}, Q3={q3:.2f}). "
                   f"{len(outliers)} outlier tasks (>{q3 + 1.5*iqr:.2f} h). "
                   f"Median: {df['Time_taken_hours'].median():.2f} h.")
        return fig, summary
    except Exception as e:
        print(f'update_box_plot error: {e}')
        return {}, ''


# ---------------------------------------------------------------------------
# Bubble chart
# ---------------------------------------------------------------------------
@app.callback(
    [Output('bubble-plot', 'figure'),
     Output('bubble-plot-summary', 'children')],
    [Input('uploaded-file-path', 'data')]
)
def update_bubble_plot(file_path):
    if not file_path:
        return {}, ''
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        df['Time_taken_hours'] = df['Time_taken'] / 60
        y_col = 'Resource' if 'Resource' in df.columns else 'Activity_type'
        hover = [c for c in ['Task_name', 'ID', 'Time_taken_hours'] if c in df.columns]
        fig = px.scatter(df, x='Timelines', y=y_col,
                         size='Time_taken_hours', color='Activity_type',
                         size_max=55, hover_data=hover,
                         title='Bubble Chart — Time Taken (Hours)')
        fig.update_layout(**make_dark_chart_layout())

        biggest = df.loc[df['Time_taken_hours'].idxmax()]
        dominant = df['Activity_type'].mode()[0]
        summary = (f"Largest task: {biggest.get('Task_name', 'N/A')} ({biggest['Time_taken_hours']:.2f} h) "
                   f"on {biggest['Timelines']}. Dominant activity: {dominant}.")
        return fig, summary
    except Exception as e:
        print(f'update_bubble_plot error: {e}')
        return {}, ''


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
@app.callback(
    [Output('heatmap', 'figure'),
     Output('heatmap-summary', 'children')],
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('uploaded-file-path', 'data')]
)
def update_heatmap(x_col, y_col, file_path):
    if not file_path:
        return {}, ''
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        if x_col not in df.columns or y_col not in df.columns:
            return {}, 'Selected column not in data.'
        df['FTE'] = df['Time_taken'] / (60 * WORK_HOURS)
        pivot = df.groupby([y_col, x_col])['FTE'].sum().reset_index().pivot(
            index=y_col, columns=x_col, values='FTE'
        )
        fig = px.imshow(pivot, labels=dict(x=x_col, y=y_col, color='FTE'),
                        color_continuous_scale='blues', aspect='auto',
                        title=f'FTE Heatmap — {y_col} vs {x_col}')
        fig.update_layout(**make_dark_chart_layout())
        fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(color='white'), title=dict(text='FTE', font=dict(color='white'))))

        flat = df.groupby([x_col, y_col])['FTE'].sum()
        max_cell = flat.idxmax()
        summary = (f"Hotspot: {max_cell[0]} / {max_cell[1]} = {flat.max():.2f} FTE. "
                   f"Total FTE in view: {flat.sum():.2f}.")
        return fig, summary
    except Exception as e:
        print(f'update_heatmap error: {e}')
        return {}, ''


# ---------------------------------------------------------------------------
# Stacked workload chart
# ---------------------------------------------------------------------------
@app.callback(
    [Output('stacked-bar-chart', 'figure'),
     Output('top-activities-table', 'data')],
    [Input('uploaded-file-path', 'data'),
     Input('threshold-line1', 'value'),
     Input('threshold-line2', 'value')]
)
def update_stacked_bar_chart(file_path, thr1, thr2):
    if not file_path:
        return {}, []
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        df['FTE'] = df['Time_taken'] / (60 * WORK_HOURS)

        grouped = df.groupby(['Timelines', 'Activity_type'])['FTE'].sum().reset_index()
        fig = px.bar(grouped, x='Timelines', y='FTE', color='Activity_type',
                     barmode='stack', title='Work Load Distribution by Work Day (FTE)',
                     labels={'Timelines': 'Work Day', 'FTE': 'FTE'})

        x_vals = grouped['Timelines'].tolist()
        for val, col, name in [(thr1, 'red', 'Capacity Line 1'), (thr2, 'orange', 'Capacity Line 2')]:
            if val:
                fig.add_hline(y=val, line_color=col, line_dash='dot',
                              annotation_text=f'{name}: {val} FTE',
                              annotation_font_color=col)

        fig.update_layout(**make_dark_chart_layout(), barmode='stack')
        fig.update_yaxes(tickformat='.2f')

        top = (df.groupby(['Timelines', 'Activity_type'])['FTE']
               .sum().reset_index()
               .sort_values(['Timelines', 'FTE'], ascending=[True, False])
               .groupby('Timelines').head(3)
               .reset_index(drop=True))
        top['FTE'] = top['FTE'].round(3)
        top.rename(columns={'FTE': 'Time_taken_FTE'}, inplace=True)
        return fig, top.to_dict('records')
    except Exception as e:
        print(f'update_stacked_bar_chart error: {e}')
        return {}, []


# ---------------------------------------------------------------------------
# Cluster tasks (the core improved callback)
# ---------------------------------------------------------------------------
@app.callback(
    [Output('level-fte', 'children'),
     Output('cluster-chart', 'figure'),
     Output('level-load-chart', 'figure'),
     Output('savings-by-cluster-chart', 'figure'),
     Output('silhouette-chart', 'figure'),
     Output('cluster-metrics', 'children'),
     Output('prediction-data-store', 'data')],
    [Input('Peak-workday', 'value'),
     Input('Activity-type', 'value'),
     Input('percentile-slider', 'value'),
     Input('use-time-feature', 'value'),
     Input('uploaded-file-path', 'data')]
)
def cluster_tasks(workday, activity, quantile_value, use_time, file_path):
    empty = ({}, {}, {}, {}, 'No data loaded yet.', {})

    if not file_path:
        return ('', *empty)

    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)

        filtered = df.copy()
        if workday:
            filtered = filtered[filtered['Timelines'] == workday]
        if activity:
            filtered = filtered[filtered['Activity_type'] == activity]

        if filtered.empty:
            return ('No data for selected filters.', *empty)

        include_time = bool(use_time)
        result, label_map, k_scores = perform_clustering(filtered, quantile_value, include_time)

        if result.empty or 'Task_cluster' not in result.columns:
            return ('Clustering failed — too few tasks (need at least 4).', *empty)

        n_clusters = result['Task_cluster'].nunique()
        time_saved_fte = round(result['Time_saved'].sum() / 60 / WORK_HOURS, 2)
        pct_level_loaded = round(100 * (result['Level_Loaded_Flag'] == 'Yes').sum() / len(result), 1)
        n_outliers = result['Is_Outlier'].sum() if 'Is_Outlier' in result.columns else 0

        best_sil = max(k_scores, key=lambda x: x['silhouette'])['silhouette'] if k_scores else 'N/A'

        label_str = (f'Scope to save {time_saved_fte:.2f} FTE by standardising similar '
                     f'{activity or "all"} tasks on {workday or "all days"}')

        # Cluster scatter
        fig1 = px.strip(result, x='Task_cluster', y='Time_taken', color='Task_cluster',
                        hover_data=[c for c in ['Task_name', 'Standard_time', 'Level_Loaded_Flag', 'Is_Outlier'] if c in result.columns],
                        title='Task Clusters — Individual Task Times (mins)')
        # Add standard time markers
        if 'Standard_time' in result.columns:
            std_df = result.groupby('Task_cluster')['Standard_time'].first().reset_index()
            fig1.add_trace(go.Scatter(
                x=std_df['Task_cluster'], y=std_df['Standard_time'],
                mode='markers', marker=dict(symbol='line-ew', size=18, color='yellow', line=dict(width=2, color='yellow')),
                name='Standard Time', showlegend=True
            ))
        fig1.update_layout(**make_dark_chart_layout())

        # Level load comparison box
        result_melted = result.copy()
        result_melted['actual'] = result_melted['Time_taken']
        result_melted['levelled'] = result_melted['Level_loaded_time']
        box_data = pd.concat([
            result_melted[['Task_cluster', 'actual']].rename(columns={'actual': 'Minutes'}).assign(Type='Actual'),
            result_melted[['Task_cluster', 'levelled']].rename(columns={'levelled': 'Minutes'}).assign(Type='Level Loaded'),
        ])
        fig2 = px.box(box_data, x='Task_cluster', y='Minutes', color='Type',
                      title='Before vs. After Level Loading by Cluster',
                      points='suspectedoutliers')
        fig2.update_layout(**make_dark_chart_layout())

        # Savings by cluster
        savings = result.groupby('Task_cluster').agg(
            FTE_Saved=('Time_saved', lambda x: round(x.sum() / 60 / WORK_HOURS, 3)),
            Tasks_Level_Loaded=('Level_Loaded_Flag', lambda x: (x == 'Yes').sum()),
        ).reset_index().sort_values('FTE_Saved', ascending=False)
        fig3 = px.bar(savings, x='Task_cluster', y='FTE_Saved',
                      color='FTE_Saved', color_continuous_scale='greens',
                      text='FTE_Saved',
                      title='FTE Savings Potential by Cluster')
        fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig3.update_layout(**make_dark_chart_layout())
        fig3.update_coloraxes(showscale=False)

        # Silhouette chart
        if k_scores:
            sil_df = pd.DataFrame(k_scores)
            fig4 = px.line(sil_df, x='k', y='silhouette', markers=True,
                           title='Silhouette Score vs Number of Clusters (K)')
            best_k_row = max(k_scores, key=lambda x: x['silhouette'])
            fig4.add_vline(x=best_k_row['k'], line_dash='dash', line_color='yellow',
                           annotation_text=f"Best K={best_k_row['k']}", annotation_font_color='yellow')
            fig4.update_layout(**make_dark_chart_layout())
        else:
            fig4 = {}

        # Metrics bar content
        metrics = html.Div(style={'display': 'flex', 'gap': '30px', 'flexWrap': 'wrap'}, children=[
            html.Span([html.B('Clusters found: '), str(n_clusters)]),
            html.Span([html.B('Silhouette score: '), str(best_sil)]),
            html.Span([html.B('Tasks level-loaded: '), f'{pct_level_loaded}%']),
            html.Span([html.B('Outlier tasks: '), str(n_outliers)]),
            html.Span([html.B('FTE saving scope: '), f'{time_saved_fte:.2f} FTE']),
        ])

        return label_str, fig1, fig2, fig3, fig4, metrics, result.to_dict('records')

    except Exception as e:
        print(f'cluster_tasks error: {e}')
        return ('Error during clustering.', *empty)


# ---------------------------------------------------------------------------
# Export level loading recommendations
# ---------------------------------------------------------------------------
@app.callback(
    Output('prediction-data-download', 'data'),
    [Input('export-button-pred', 'n_clicks')],
    [State('prediction-data-store', 'data')],
    prevent_initial_call=True
)
def export_level_loading(n_clicks, store_data):
    if not n_clicks or not store_data:
        return None
    df = pd.DataFrame(store_data)
    temp_dir = os.path.join(base_path, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, 'level_loading_recommendations.csv')
    df.to_csv(path, index=False)
    temp_files.append(path)
    return dcc.send_file(path)


# ---------------------------------------------------------------------------
# Export top 3 workdays & activities
# ---------------------------------------------------------------------------
@app.callback(
    Output('multi-data-download', 'data'),
    [Input('export-multi-button', 'n_clicks')],
    [State('uploaded-file-path', 'data')],
    prevent_initial_call=True
)
def export_multi_recommendations(n_clicks, file_path):
    if not n_clicks or not file_path:
        return None
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        df['FTE'] = df['Time_taken'] / (60 * WORK_HOURS)

        top_days = df.groupby('Timelines')['FTE'].sum().nlargest(3).index.tolist()
        all_results = []
        for day in top_days:
            daily = df[df['Timelines'] == day]
            top_acts = daily.groupby('Activity_type')['FTE'].sum().nlargest(3).index.tolist()
            for act in top_acts:
                sub = daily[daily['Activity_type'] == act].copy()
                enriched, _, _ = perform_clustering(sub, quantile_value=75, use_time_feature=True)
                enriched['Selected_Timelines'] = day
                enriched['Selected_Activity_type'] = act
                all_results.append(enriched)

        final = pd.concat(all_results, ignore_index=True)
        temp_dir = os.path.join(base_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, 'top3_recommendations.csv')
        final.to_csv(path, index=False)
        temp_files.append(path)
        return dcc.send_file(path)
    except Exception as e:
        print(f'export_multi_recommendations error: {e}')
        return None


# ---------------------------------------------------------------------------
# Input sheet template
# ---------------------------------------------------------------------------
@app.callback(
    Output('input-sheet-download', 'data'),
    [Input('input-sheet', 'n_clicks')],
    prevent_initial_call=True
)
def export_input_template(n_clicks):
    if not n_clicks:
        return None
    template = pd.DataFrame(columns=['Timelines', 'Activity_type', 'Task_name',
                                      'Time_taken', 'Resource', 'Entity', 'Company_code'])
    temp_dir = os.path.join(base_path, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, 'orbit_input_template.csv')
    template.to_csv(path, index=False)
    temp_files.append(path)
    return dcc.send_file(path)


# ---------------------------------------------------------------------------
# Action table export
# ---------------------------------------------------------------------------
@app.callback(
    Output('data-download', 'data'),
    [Input('export-button', 'n_clicks')],
    [State('data-table', 'data')],
    prevent_initial_call=True
)
def export_action_table(n_clicks, table_data):
    if not n_clicks or not table_data:
        return None
    df = pd.DataFrame(table_data)
    temp_dir = os.path.join(base_path, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, 'orbit_actions.csv')
    df.to_csv(path, index=False)
    temp_files.append(path)
    return dcc.send_file(path)


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) { window.close(); }
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    [Input('exit-button', 'n_clicks')]
)

@app.callback(
    Output('exit-button', 'disabled'),
    [Input('exit-button', 'n_clicks')]
)
def handle_server_exit(n_clicks):
    if n_clicks and n_clicks > 0:
        cleanup_temp_files()
        # Use a background thread so the response can be sent first
        def _shutdown():
            import time
            time.sleep(0.5)
            os._exit(0)
        from threading import Thread
        Thread(target=_shutdown, daemon=True).start()
    return False


# ---------------------------------------------------------------------------
# Resource Utilization — overview (heatmap, bar, DC/Entity bar, overload alert)
# ---------------------------------------------------------------------------
@app.callback(
    [Output('resource-heatmap', 'figure'),
     Output('dc-entity-bar', 'figure'),
     Output('resource-activity-selector', 'options'),
     Output('resource-overload-alert', 'children')],
    [Input('uploaded-file-path', 'data')]
)
def update_resource_overview(file_path):
    empty = ({}, {}, [], '')
    if not file_path:
        return empty
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        df['FTE'] = df['Time_taken'] / (60 * WORK_HOURS)

        has_resource = 'Resource' in df.columns

        # ── Heatmap: Resource × Timelines ──
        if has_resource:
            pivot = (df.groupby(['Resource', 'Timelines'])['FTE']
                     .sum().reset_index()
                     .pivot(index='Resource', columns='Timelines', values='FTE')
                     .fillna(0))
            fig_heat = px.imshow(
                pivot,
                labels=dict(x='Work Day', y='Resource', color='FTE'),
                color_continuous_scale='reds',
                aspect='auto',
                title='Resource × Work Day Heatmap (FTE)',
            )
            fig_heat.update_layout(**make_dark_chart_layout())
            fig_heat.update_layout(
                coloraxis_colorbar=dict(
                    tickfont=dict(color='white'),
                    title=dict(text='FTE', font=dict(color='white')),
                )
            )
        else:
            fig_heat = go.Figure()
            fig_heat.update_layout(**make_dark_chart_layout(title='Resource column not in data'))

        # ── Bar: Delivery Centre / Entity split ──
        group_col = next((c for c in ['Delivery_Centre', 'Entity', 'Company_code']
                          if c in df.columns), None)
        if group_col:
            dc_df = (df.groupby([group_col, 'Activity_type'])['FTE']
                     .sum().reset_index()
                     .sort_values('FTE', ascending=False))
            fig_dc = px.bar(
                dc_df, x=group_col, y='FTE', color='Activity_type',
                barmode='stack',
                title=f'FTE by {group_col} and Activity Type',
            )
            fig_dc.update_layout(**make_dark_chart_layout())
        else:
            fig_dc = go.Figure()
            fig_dc.update_layout(**make_dark_chart_layout(
                title='No Delivery Centre / Entity / Company Code column found'))

        # ── Activity selector options ──
        act_opts = [{'label': a, 'value': a}
                    for a in sorted(df['Activity_type'].dropna().unique())]

        # ── Overload alert ──
        alert = ''
        if has_resource:
            daily_res = df.groupby(['Resource', 'Timelines'])['FTE'].sum()
            overloaded = daily_res[daily_res > 1.0]
            if not overloaded.empty:
                names = overloaded.index.get_level_values('Resource').unique().tolist()
                alert = (f"⚠ Overloaded resources (>1.0 FTE on at least one day): "
                         f"{', '.join(names)}")

        return fig_heat, fig_dc, act_opts, alert

    except Exception as e:
        print(f'update_resource_overview error: {e}')
        return {}, {}, [], ''


# ---------------------------------------------------------------------------
# Resource Utilization — task-duration comparison per resource
# ---------------------------------------------------------------------------
@app.callback(
    Output('resource-task-box', 'figure'),
    [Input('resource-activity-selector', 'value'),
     Input('uploaded-file-path', 'data')]
)
def update_resource_task_comparison(activity, file_path):
    if not file_path or not activity:
        return {}
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        if 'Resource' not in df.columns:
            return {}
        sub = df[df['Activity_type'] == activity].copy()
        sub['Time_taken_hours'] = sub['Time_taken'] / 60
        hover = [c for c in ['Task_name', 'Timelines', 'Entity'] if c in sub.columns]
        fig = px.box(
            sub, x='Resource', y='Time_taken_hours',
            points='all', hover_data=hover,
            title=f'Task Duration by Resource — {activity}',
            color='Resource',
        )
        fig.update_layout(**make_dark_chart_layout())
        # Overlay cluster mean line
        mean_val = sub['Time_taken_hours'].mean()
        fig.add_hline(y=mean_val, line_dash='dash', line_color='yellow',
                      annotation_text=f'Overall mean: {mean_val:.2f} h',
                      annotation_font_color='yellow')
        return fig
    except Exception as e:
        print(f'update_resource_task_comparison error: {e}')
        return {}


# ---------------------------------------------------------------------------
# What-If Simulator — populate dropdowns from file
# ---------------------------------------------------------------------------
@app.callback(
    [Output('whatif-source-day', 'options'),
     Output('whatif-target-days', 'options'),
     Output('whatif-activity', 'options')],
    [Input('uploaded-file-path', 'data')]
)
def populate_whatif_dropdowns(file_path):
    if not file_path:
        return [], [], []
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        df['FTE'] = df['Time_taken'] / (60 * WORK_HOURS)

        # Sort days by total FTE descending so the peak appears first
        day_fte = df.groupby('Timelines')['FTE'].sum().sort_values(ascending=False)
        day_opts = [{'label': f"{d}  ({day_fte[d]:.1f} FTE)", 'value': d}
                    for d in day_fte.index]
        act_opts = [{'label': a, 'value': a}
                    for a in sorted(df['Activity_type'].dropna().unique())]
        return day_opts, day_opts, act_opts
    except Exception as e:
        print(f'populate_whatif_dropdowns error: {e}')
        return [], [], []


# ---------------------------------------------------------------------------
# What-If Simulator — update slider label
# ---------------------------------------------------------------------------
@app.callback(
    Output('whatif-slider-label', 'children'),
    [Input('whatif-pct', 'value'),
     Input('whatif-activity', 'value'),
     Input('whatif-source-day', 'value')]
)
def update_whatif_label(pct, activity, source):
    act_str = activity or 'all activities'
    src_str = source or '(source day)'
    return f'Shift {pct}% of {act_str} from {src_str}'


# ---------------------------------------------------------------------------
# What-If Simulator — main chart + headline + task table
# ---------------------------------------------------------------------------
@app.callback(
    [Output('whatif-chart', 'figure'),
     Output('whatif-headline', 'children'),
     Output('whatif-task-table', 'data')],
    [Input('whatif-source-day', 'value'),
     Input('whatif-target-days', 'value'),
     Input('whatif-activity', 'value'),
     Input('whatif-pct', 'value'),
     Input('uploaded-file-path', 'data')]
)
def update_whatif(source_day, target_days, activity, pct, file_path):
    no_result = ({}, 'Select a source day and at least one target day to run the simulation.', [])
    if not file_path or not source_day or not target_days:
        return no_result
    try:
        df = pd.read_csv(file_path)
        df['Time_taken'] = pd.to_numeric(df['Time_taken'], errors='coerce')
        df.dropna(subset=['Time_taken'], inplace=True)
        df['FTE'] = df['Time_taken'] / (60 * WORK_HOURS)

        # Ensure target list doesn't include the source
        targets = [d for d in target_days if d != source_day]
        if not targets:
            return ({}, 'Target day(s) must differ from the source day.', [])

        shift_frac = pct / 100.0

        # ── Identify the tasks to shift ──
        mask = df['Timelines'] == source_day
        if activity:
            mask &= df['Activity_type'] == activity

        source_tasks = df[mask].copy()
        if source_tasks.empty:
            return ({}, f'No tasks found on {source_day} for the selected activity.', [])

        # Sort longest tasks first; take the top fraction by cumulative FTE
        source_tasks = source_tasks.sort_values('FTE', ascending=False)
        total_source_fte = source_tasks['FTE'].sum()
        target_fte = total_source_fte * shift_frac

        # Select tasks greedily until we reach the target FTE
        source_tasks['cum_FTE'] = source_tasks['FTE'].cumsum()
        tasks_to_move = source_tasks[source_tasks['cum_FTE'] <= target_fte].copy()
        # Always move at least one task
        if tasks_to_move.empty:
            tasks_to_move = source_tasks.iloc[:1].copy()

        moved_fte = tasks_to_move['FTE'].sum()
        n_moved = len(tasks_to_move)

        # ── Build simulated daily totals ──
        daily = df.groupby('Timelines')['FTE'].sum().reset_index().copy()
        daily.columns = ['Timelines', 'Actual_FTE']
        daily['Simulated_FTE'] = daily['Actual_FTE'].copy()

        # Deduct from source
        src_idx = daily['Timelines'] == source_day
        daily.loc[src_idx, 'Simulated_FTE'] -= moved_fte

        # Distribute equally across targets
        per_target = moved_fte / len(targets)
        for t in targets:
            tgt_idx = daily['Timelines'] == t
            if tgt_idx.any():
                daily.loc[tgt_idx, 'Simulated_FTE'] += per_target
            else:
                # Target day not yet in data — add a new row
                new_row = pd.DataFrame([{
                    'Timelines': t,
                    'Actual_FTE': 0,
                    'Simulated_FTE': per_target
                }])
                daily = pd.concat([daily, new_row], ignore_index=True)

        # ── Chart ──
        fig = go.Figure()
        fig.add_bar(name='Actual', x=daily['Timelines'], y=daily['Actual_FTE'],
                    marker_color='rgba(100,150,255,0.7)',
                    text=daily['Actual_FTE'].round(2),
                    texttemplate='%{text:.2f}', textposition='outside')
        fig.add_bar(name='Simulated', x=daily['Timelines'], y=daily['Simulated_FTE'],
                    marker_color='rgba(100,255,160,0.7)',
                    text=daily['Simulated_FTE'].round(2),
                    texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            **make_dark_chart_layout(title='Load Curve — Actual vs. Simulated (FTE)'),
            barmode='group',
        )

        # ── Headline ──
        old_peak = daily['Actual_FTE'].max()
        new_peak = daily['Simulated_FTE'].max()
        old_peak_day = daily.loc[daily['Actual_FTE'].idxmax(), 'Timelines']
        new_peak_day = daily.loc[daily['Simulated_FTE'].idxmax(), 'Timelines']
        delta = old_peak - new_peak
        act_str = activity or 'all activities'

        if delta > 0.001:
            headline = (
                f"Shifting {pct}% of '{act_str}' from {source_day} to "
                f"{', '.join(targets)} moves {n_moved} task(s) ({moved_fte:.2f} FTE). "
                f"Peak load drops from {old_peak:.2f} FTE ({old_peak_day}) "
                f"→ {new_peak:.2f} FTE ({new_peak_day})  —  saving {delta:.2f} FTE on peak day."
            )
        else:
            headline = (
                f"Shifting {pct}% of '{act_str}' moves {n_moved} task(s) ({moved_fte:.2f} FTE) "
                f"but the overall peak remains at {new_peak:.2f} FTE (peak shifts to {new_peak_day})."
            )

        # ── Task table ──
        table_cols = ['Task_name', 'Activity_type', 'Time_taken']
        if 'Resource' in tasks_to_move.columns:
            table_cols.insert(2, 'Resource')
        export = tasks_to_move[table_cols].copy()
        export['Time_taken_hrs'] = (export['Time_taken'] / 60).round(2)
        export['FTE_contribution'] = tasks_to_move['FTE'].round(3).values
        export['Time_taken'] = export['Time_taken'].round(0).astype(int)

        return fig, headline, export.to_dict('records')

    except Exception as e:
        print(f'update_whatif error: {e}')
        return {}, 'Error running simulation.', []


# ===========================================================================
# Entry point
# ===========================================================================
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    _, port = s.getsockname()
    s.close()
    return port


if __name__ == '__main__':
    port = find_free_port()
    Timer(1.2, lambda: webbrowser.open_new_tab(f'http://127.0.0.1:{port}/')).start()
    app.run(port=port, debug=False)
