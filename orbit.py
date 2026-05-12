import socket
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
from io import BytesIO
import pandas as pd
import dash
from dash import html, dcc, dash_table
import webbrowser
from threading import Timer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import filedialog
import plotly.express as px
import dash_auth
import base64
import os
import atexit
import signal
import sys

#credentials
VALID_USERNAME_PASSWORD_PAIRS = {
    'Orbit': '123gen'
}

table_data = pd.DataFrame(columns=['Insight', 'Action', 'Owner', 'Target Date'])
empty_rows = pd.DataFrame('', index=range(5), columns=table_data.columns)
table_data = pd.concat([table_data, empty_rows], ignore_index=True)

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
app.title = 'Orbit 1.0'

temp_files = []

def cleanup_temp_files():
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Deleted temporary file: {temp_file}")
        except Exception as e:
            print(f"Error deleting temporary file {temp_file}: {e}")

atexit.register(cleanup_temp_files)

def exit_app():
    print("Exiting the application...")
    cleanup_temp_files()
    os.kill(os.getpid(), signal.SIGTERM)


if getattr(sys, 'frozen', False):  
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

assets_path = r'assets\OrbitLogo.png'
templates_path = os.path.join(base_path, 'templates')


app.layout = html.Div(style={
    'background': 'linear-gradient(to left, #000033, #000066, #000099)',
    'justify-content': 'center',
    'color': 'white',
    'font-family': 'Roboto, sans-serif',
    'padding': '20px',
    'height': '100vh'
}, children=[
    html.Div([
        html.Img(src=os.path.join(assets_path), style={'height': '100px', 'width': 'auto', 'margin': '0px'})
    ], style={'width': '100%', 'text-align': 'center', 'display': 'inline-block'}),
    html.Div([
        html.B("ORBIT 1.2", style={'fontSize': '20px'}),
        html.Hr(),
        html.P("Use these visualizations to understand the distribution of work with various parameters",
               style={'fontSize': '20px'}),
    ], style={'width': '100%', 'text-align': 'center', 'display': 'inline-block'}),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV/Excel File', style={
                'backgroundColor': '#007bff',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'cursor': 'pointer'
            }),
            multiple=False
        ),
        html.Div(id='file-upload-status', style={
            'marginTop': '10px',
            'fontSize': '16px',
            'color': 'lightgreen'
        })
    ], style={'marginBottom': '20px'}),
    dcc.Store(id='uploaded-file-path', storage_type='memory'),
    dcc.Tabs([
        dcc.Tab(label='Orbit Data Mining', children=[
            dcc.Tabs([
                dcc.Tab(label='Bar Chart', children=[
                    html.Div([
                        html.Br(),
                        html.H1(id='fte', style={'font-family': 'Calibri', 'fontSize': '30px'}),
                        html.Label('Select X-Axis:', style={'font-family': 'Calibri', 'fontSize': '15px'}),
                        dcc.Dropdown(
                            id='x-axis-column',
                            options=[
                                {'label': 'Cluster', 'value': 'Cluster'},
                                {'label': 'Delivery Centre', 'value': 'Delivery_Centre'},
                                {'label': 'Activity Type', 'value': 'Activity_type'},
                                {'label': 'Timelines', 'value': 'Timelines'},
                                {'label': 'Resource', 'value': 'Resource'}
                            ],
                            placeholder="Select X-Axis for Bar Graph",
                            value='Cluster'
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    dcc.Graph(id='bar-chart'),
                    html.Div(id='bar-chart-summary', style={
                        'marginTop': '10px',
                        'fontSize': '14px',
                        'color': 'lightblue'
                    }),
                ]),
                dcc.Tab(label='Box Plot', children=[
                    html.Div([
                        html.Label('Select X-Axis:', style={'font-family': 'Calibri', 'fontSize': '15px'}),
                        dcc.Dropdown(
                            id='multi-x-axis-columns',
                            options=[
                                {'label': 'Entity', 'value': 'Entity'},
                                {'label': 'Company Code', 'value': 'Company_code'},
                                {'label': 'Resource', 'value': 'Resource'}
                            ],
                            placeholder="Select X-Axis for Box Plot",
                            value='Entity'
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    dcc.Graph(id='box-plot'),
                    html.Div(id='box-plot-summary', style={
                        'marginTop': '10px',
                        'fontSize': '14px',
                        'color': 'lightblue'
                    }),
                ]),
                dcc.Tab(label='Bubble Chart', children=[
                    dcc.Graph(id='bubble-plot'),
                    html.Div(id='bubble-plot-summary', style={
                        'marginTop': '10px',
                        'fontSize': '14px',
                        'color': 'lightblue'
                    }),
                ]),
                dcc.Tab(label='Heatmap', children=[
                    html.Div([
                        html.Label('Select X-Axis:', style={'font-family': 'Calibri', 'fontSize': '15px'}),
                        dcc.Dropdown(
                            id='x-axis-dropdown',
                            options=[
                                {'label': 'Entity', 'value': 'Entity'},
                                {'label': 'Activity Type', 'value': 'Activity_type'},
                                {'label': 'Resource', 'value': 'Resource'},
                                {'label': 'Timelines', 'value': 'Timelines'}
                            ],
                            value='Entity'
                        ),
                        html.Label('Select Y-Axis:', style={'font-family': 'Calibri', 'fontSize': '15px'}),
                        dcc.Dropdown(
                            id='y-axis-dropdown',
                            options=[
                                {'label': 'Entity', 'value': 'Entity'},
                                {'label': 'Activity Type', 'value': 'Activity_type'},
                                {'label': 'Resource', 'value': 'Resource'},
                                {'label': 'Timelines', 'value': 'Timelines'}
                            ],
                            value='Activity_type'
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    dcc.Graph(id='heatmap'),
                    html.Div(id='heatmap-summary', style={
                        'marginTop': '10px',
                        'fontSize': '14px',
                        'color': 'lightblue'
                    }),
                ]),
                dcc.Tab(label='Stacked Workload Chart', children=[
                    html.Div([
                        html.Label('Threshold Line 1 (Red):', style={'font-family': 'Calibri', 'fontSize': '15px'}),
                        dcc.Input(id='threshold-line1', type='number', value=0, step=1,
                                  style={'width': '100%', 'margin-bottom': '10px'}),
                        html.Label('Threshold Line 2 (Orange):', style={'font-family': 'Calibri', 'fontSize': '15px'}),
                        dcc.Input(id='threshold-line2', type='number', value=0, step=1,
                                  style={'width': '100%', 'margin-bottom': '10px'})
                    ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),

                    html.Div([
                        dcc.Graph(id='stacked-bar-chart')
                    ], style={'width': '78%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        html.H4("Top 3 Activities by FTE per Work Day", style={'color': 'white', 'font-family': 'Calibri'}),
                        dash_table.DataTable(
                            id='top-activities-table',
                            columns=[
                                {'name': 'Work Day', 'id': 'Timelines'},
                                {'name': 'Activity Type', 'id': 'Activity_type'},
                                {'name': 'Time Taken (FTE)', 'id': 'Time_taken_FTE'}
                            ],
                            data=[],
                            style_table={'overflowX': 'auto', 'backgroundColor': 'white', 'color': 'black'},
                            style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold', 'color': 'black'},
                            style_cell={'textAlign': 'center', 'color': 'black', 'font-family': 'Calibri'},
                            page_size=10
                        )
                    ], style={'marginTop': '30px', 'padding': '10px'})
                ])
            ])
        ]),
        dcc.Tab(label='Task Clusters & Action Table', children=[
            html.Div([
                html.H3('Find Clusters of Tasks that can be standardized', style={'fontSize': '20px'}),
                html.Div([
                    html.Label('Select Peak Workday:',
                               style={'font-family': 'Calibri', 'fontSize': '15px'}),
                    dcc.Dropdown(id='Peak-workday', options=[], multi=False),
                    html.Label('Select Activity Type to Level Load:',
                               style={'font-family': 'Calibri', 'fontSize': '15px'}),
                    dcc.Dropdown(id='Activity-type', options=[], multi=False),
                    html.Label('How Aggressively do you want to Standardize:',
                               style={'font-family': 'Calibri', 'fontSize': '15px'}),
                    dcc.Slider(id='percentile-slider', min=75, max=90, value=75, step=5,
                               marks={75: {'label': 'Aggressive', 'style': {'color': '#fff'}},
                                      90: {'label': 'Lenient', 'style': {'color': '#fff'}}}),
                ], style={'width': '30%', 'display': 'inline-block'}),
                html.H1(id='level-fte', style={'font-family': 'Calibri', 'fontSize': '30px'}),
                dcc.Graph(id='cluster-chart'),
                dcc.Graph(id='level-load-chart'),
                html.Button('Export Level Loading Recommendations', id='export-button-pred', n_clicks=0),
                dcc.Store(id='prediction-data-store'),
                dcc.Download(id='prediction-data-download'),
                html.Button('Export Top 3 Workdays & Activities', id='export-multi-button', n_clicks=0),
                dcc.Download(id='multi-data-download'),
                html.Button('Input Sheet for Recommendations', id='input-sheet', n_clicks=0),
                dcc.Download(id='input-sheet-download')
            ]),
            html.Br(),
            html.Div([
                html.H3('Fill Actionables Basis Insights', style={'fontSize': '20px'}),
                dash_table.DataTable(
                    id='data-table',
                    columns=[{'name': col, 'id': col, 'editable': True} for col in table_data.columns],
                    data=table_data.to_dict('records'),
                    style_table={'overflowX': 'auto', 'backgroundColor': 'transparent'},
                    style_header={'backgroundColor': '#333', 'color': 'white', 'textAlign': 'center'},
                    style_cell={
                        'textAlign': 'center',
                        'backgroundColor': 'transparent',
                        'color': 'white',
                        'border': '1px solid white'
                    }
                ),
                html.Button('Export to Excel', id='export-button', n_clicks=0),
                dcc.Download(id='data-download')
            ])
        ])
    ]),
    html.Div([
        html.Button('Exit Application', id='exit-button', n_clicks=0,
                    style={'backgroundColor': 'red', 'color': 'white', 'border': 'none', 'padding': '10px 20px',
                           'cursor': 'pointer', 'marginTop': '20px'})
    ], style={'textAlign': 'center', 'width': '100%'}),
    html.Div(id='dummy-output', style={'display': 'none'})
])

def perform_clustering(df, quantile_value):
    """
    Perform task clustering and leveling logic.
    Returns a DataFrame with recommendation data.
    """
    if df.empty:
        return pd.DataFrame()

    try:
        tasks = df['Task_name'].dropna().tolist()
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(tasks)
        max_clusters = min(10, len(tasks))
        kmeans = KMeans(n_clusters=max_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        cluster_names = [f"{df['Activity_type'].iloc[i]} {chr(65 + i % 26)}" for i in range(max_clusters)]
        df.loc[:, 'Task_cluster'] = [cluster_names[label] for label in cluster_labels]

        quantile_proportion = quantile_value / 100.0
        cluster_avg_times = df.groupby('Task_cluster')['Time_taken'].quantile(quantile_proportion).reset_index()
        cluster_avg_times.rename(columns={'Time_taken': 'Avg_Time_taken'}, inplace=True)

        df = pd.merge(df, cluster_avg_times, on='Task_cluster', how='left')
        df['Level_loaded_time'] = df.apply(
            lambda row: row['Avg_Time_taken'] if row['Time_taken'] > row['Avg_Time_taken'] else row['Time_taken'],
            axis=1
        )
        df['Level_Loaded_Flag'] = df.apply(lambda row: 'Yes' if row['Level_loaded_time'] != row['Time_taken'] else 'No', axis=1)
        df['Time_saved'] = df['Time_taken'] - df['Level_loaded_time']
        df['Time_taken_FTE'] = df['Time_taken'] / (60 * 8.50)
        df['Level_loaded_FTE'] = df['Level_loaded_time'] / (60 * 8.50)

        return df
    except Exception as e:
        print(f"Error in perform_clustering: {e}")
        return pd.DataFrame()


@app.callback(
    [dash.dependencies.Output('uploaded-file-path', 'data'),
     dash.dependencies.Output('file-upload-status', 'children')],
    [dash.dependencies.Input('upload-data', 'contents')],
    [dash.dependencies.State('upload-data', 'filename')]
)
def handle_file_upload(contents, filename):
    if contents is None:
        return None, "No file uploaded yet."
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        temp_dir = os.path.join(base_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{filename}")
        with open(temp_file_path, 'wb') as f:
            f.write(decoded)
        temp_files.append(temp_file_path)
        return temp_file_path, f"File '{filename}' uploaded successfully!"
    except Exception as e:
        print(f"Error in file upload: {e}")
        return None, "Error: Unable to process the uploaded file."

app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            window.close();  // Close the browser tab
        }
        return '';
    }
    """,
    dash.dependencies.Output('dummy-output', 'children'),
    [dash.dependencies.Input('exit-button', 'n_clicks')]
)

@app.callback(
    [dash.dependencies.Output('Peak-workday', 'options'),
     dash.dependencies.Output('Activity-type', 'options')],
    [dash.dependencies.Input('uploaded-file-path', 'data')]
)
def update_dropdowns(uploaded_file_path):
    file_path = uploaded_file_path
    if not file_path:
        return [], []
    try:
        df = pd.read_csv(file_path)
        timeline_options = [{'label': t, 'value': t} for t in df['Timelines'].unique()]
        activity_type_options = [{'label': t, 'value': t} for t in df['Activity_type'].unique()]
        return timeline_options, activity_type_options
    except Exception as e:
        print(f"Error in Dropdown Update Callback: {e}")
        return [], []

@app.callback(
    [dash.dependencies.Output('fte', 'children'), dash.dependencies.Output('bar-chart', 'figure'),
     dash.dependencies.Output('bar-chart-summary', 'children')],
    [dash.dependencies.Input('x-axis-column', 'value'),
     dash.dependencies.Input('uploaded-file-path', 'data')]
)
def update_bar_chart(x_axis_column, uploaded_file_path):
    file_path = uploaded_file_path
    if not file_path:
        return "Error: File path not set.", {}, "No data available."
    try:
        df = pd.read_csv(file_path)
        if 'Timelines' not in df.columns or 'Time_taken' not in df.columns:
            return "Error: Required columns (Timelines, Time_taken) are missing.", {}, "Error: Missing required columns."
        df.dropna(subset=['Time_taken'], inplace=True)
        df['Time_taken_hours'] = df['Time_taken'] / 60
        grouped_peak = df.groupby('Timelines')['Time_taken_hours'].sum().reset_index()
        grouped_peak_sorted = grouped_peak.sort_values(by='Time_taken_hours', ascending=False)
        highest_value_grouped = grouped_peak_sorted['Time_taken_hours'].max()
        fte = round((highest_value_grouped / 8.50), 0)
        fte_string = f"Peak Load = {fte:.2f} FTE"
        total_time_by_entity = df.groupby(x_axis_column)['Time_taken_hours'].sum().reset_index()
        fig = px.bar(total_time_by_entity, x=x_axis_column, y='Time_taken_hours',
                     color='Time_taken_hours', color_continuous_scale='sunsetdark',
                     title=f'Total Time Taken per {x_axis_column} (Hours)', text='Time_taken_hours')
        fig.update_xaxes(title_text=x_axis_column)
        fig.update_yaxes(title_text='Total Time Taken (Hours)')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        data_description = (
            f"The bar chart shows the total time taken per {x_axis_column}. "
            f"The maximum time taken is {total_time_by_entity['Time_taken_hours'].max():.2f} hours, "
            f"the minimum time taken is {total_time_by_entity['Time_taken_hours'].min():.2f} hours, "
            f"and the average time taken is {total_time_by_entity['Time_taken_hours'].mean():.2f} hours. "
            f"The data is grouped by {x_axis_column}, and the values represent the sum of time taken in hours."
        )
        summary_text = data_description

        return fte_string, fig, summary_text
    except Exception as e:
        print(f"Error in Bar Chart Callback: {e}")
        return "Error: Unable to generate bar chart.", {}, "Error: Unable to generate summary."

@app.callback(
    [dash.dependencies.Output('level-fte', 'children'), dash.dependencies.Output('cluster-chart', 'figure'),
     dash.dependencies.Output('level-load-chart', 'figure'), dash.dependencies.Output('prediction-data-store', 'data')],
    [dash.dependencies.Input('Peak-workday', 'value'),
     dash.dependencies.Input('Activity-type', 'value'),
     dash.dependencies.Input('percentile-slider', 'value'),
     dash.dependencies.Input('export-button-pred', 'n_clicks')],
    [dash.dependencies.State('uploaded-file-path', 'data')]
)
def cluster_tasks(filter_workday, filter_activity, quantile_value, n_clicks, uploaded_file_path):
    file_path = uploaded_file_path
    if not file_path:
        return "Error: File path not set.", {}, {}, {}
    try:
        df = pd.read_csv(file_path)
        if 'Timelines' not in df.columns or 'Activity_type' not in df.columns or 'Time_taken' not in df.columns:
            return "Error: Required columns (Timelines, Activity_type, Time_taken) are missing.", {}, {}, {}
        df.dropna(subset=['Time_taken'], inplace=True)
        filtered_df = df.copy()
        if filter_workday:
            filtered_df = filtered_df[filtered_df['Timelines'] == filter_workday]
        if filter_activity:
            filtered_df = filtered_df[filtered_df['Activity_type'] == filter_activity]
        if filtered_df.empty:
            return "Error: No data available for the selected filters.", {}, {}, {}
        tasks = filtered_df['Task_name'].dropna().tolist()
        if len(tasks) == 0:
            return "Error: No valid tasks found for clustering.", {}, {}, {}
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(tasks)
        max_clusters = 10
        kmeans = KMeans(n_clusters=max_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        cluster_names = [f"{filtered_df['Activity_type'].iloc[i]} {chr(65 + i)}" for i in range(max_clusters)]
        filtered_df.loc[:, 'Task_cluster'] = [cluster_names[label] for label in cluster_labels]
        quantile_proportion = quantile_value / 100.0
        cluster_avg_times = filtered_df.groupby('Task_cluster')['Time_taken'].quantile(quantile_proportion).reset_index()
        cluster_avg_times.rename(columns={'Time_taken': 'Avg_Time_taken'}, inplace=True)
        filtered_df = pd.merge(filtered_df, cluster_avg_times, on='Task_cluster', how='left')
        filtered_df['Level_loaded_time'] = filtered_df.apply(
            lambda row: row['Avg_Time_taken'] if row['Time_taken'] > row['Avg_Time_taken'] else row['Time_taken'],
            axis=1)
        filtered_df['Level_Loaded_Flag'] = filtered_df.apply(lambda row: 'Yes' if
        row['Level_loaded_time'] != row['Time_taken'] else 'No', axis=1)
        filtered_df['Time_saved'] = filtered_df['Time_taken'] - filtered_df['Level_loaded_time']
        time_saved_sum = round((filtered_df['Time_saved'].sum()) / 60 / 8.50, 2)
        level_fte = f"Peak Load Reduction scope of {time_saved_sum:.2f} FTE through Standardizing similar tasks on {filter_workday} {filter_activity}"
        fig1 = px.scatter(filtered_df, x='Task_cluster', y='Time_taken', color='Task_cluster',
                          hover_data=['Task_name', 'Time_taken', 'Level_loaded_time'])
        fig2 = px.box(filtered_df, x='Task_cluster', y='Time_taken', points="all", boxmode='group',
                      hover_data=['Task_name', 'Time_taken', 'Level_loaded_time'])
        return level_fte, fig1, fig2, filtered_df.to_dict(orient='records')
    except Exception as e:
        print(f"Error in Task Cluster Callback: {e}")
        return "Error: Unable to generate task clusters.", {}, {}, {}

@app.callback(
    dash.dependencies.Output('heatmap', 'figure'),
    [dash.dependencies.Input('x-axis-dropdown', 'value'),
     dash.dependencies.Input('y-axis-dropdown', 'value'),
     dash.dependencies.Input('uploaded-file-path', 'data')]
)
def update_heatmap(x_axis, y_axis, uploaded_file_path):
    file_path = uploaded_file_path
    if not file_path:
        return {}
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Time_taken'], inplace=True)
        df['Time_taken_hours'] = df['Time_taken'] / 60
        heatmap_data = df.groupby([x_axis, y_axis])['Time_taken_hours'].sum().reset_index().pivot(
            index=y_axis, columns=x_axis, values='Time_taken_hours')
        fig = px.imshow(heatmap_data, labels=dict(x=x_axis, y=y_axis),
                        color_continuous_scale='sunsetdark', aspect='auto', title='Heatmap of Time Taken (Hours)')
        fig.update_layout(margin=dict(l=50, r=50, b=50, t=50))
        return fig
    except Exception as e:
        print(f"Error in Heatmap Callback: {e}")
        return {}

@app.callback(
    [dash.dependencies.Output('box-plot', 'figure'),
     dash.dependencies.Output('box-plot-summary', 'children')],
    [dash.dependencies.Input('multi-x-axis-columns', 'value'),
     dash.dependencies.Input('uploaded-file-path', 'data')]
)
def update_box_plot(selected_columns, uploaded_file_path):
    file_path = uploaded_file_path
    if not file_path or not selected_columns:
        return {}, "No data available."
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Time_taken'], inplace=True)
        df['Time_taken_hours'] = df['Time_taken'] / 60
        fig = px.box(df, x=selected_columns, y='Time_taken_hours', title='Box Plot of Time Taken (Hours)',
                     hover_data=['Task_name', 'ID', 'Time_taken', 'Timelines'])
        fig.update_traces(marker_color='purple', marker=dict(outliercolor='red'))
        fig.update_xaxes(title_text=selected_columns)
        fig.update_yaxes(title_text='Time Taken (Hours)', range=[0, 10])

        q1 = df['Time_taken_hours'].quantile(0.25)
        q3 = df['Time_taken_hours'].quantile(0.75)
        iqr = q3 - q1
        data_description = (
            f"The box plot shows the distribution of time taken across {selected_columns}. "
            f"The interquartile range (IQR) is {iqr:.2f} hours, with Q1 at {q1:.2f} hours and Q3 at {q3:.2f} hours. "
            f"The data represents the time taken in hours, grouped by {selected_columns}."
        )

        summary_text = data_description

        return fig, summary_text
    except Exception as e:
        print(f"Error in Box Plot Callback: {e}")
        return {}, "Error: Unable to generate summary."

@app.callback(
    [dash.dependencies.Output('bubble-plot', 'figure'),
     dash.dependencies.Output('bubble-plot-summary', 'children')],
    [dash.dependencies.Input('bubble-plot', 'id'),
     dash.dependencies.Input('uploaded-file-path', 'data')]
)
def update_bubble_plot(_, uploaded_file_path):
    file_path = uploaded_file_path
    if not file_path:
        return {}, "No data available."
    try:
        dff = pd.read_csv(file_path)
        dff.dropna(subset=['Time_taken'], inplace=True)
        dff['Time_taken_hours'] = dff['Time_taken'] / 60
        fig_bubble = px.scatter(dff, x='Timelines', y='Resource', size='Time_taken_hours', color='Activity_type',
                                size_max=60, hover_data=['Task_name', 'ID', 'Time_taken_hours', 'Timelines'],
                                title='Bubble Chart of Time Taken (Hours)')

        largest_bubble = dff.loc[dff['Time_taken_hours'].idxmax()]
        smallest_bubble = dff.loc[dff['Time_taken_hours'].idxmin()]
        dominant_activity = dff['Activity_type'].mode()[0]
        data_description = (
            f"The bubble chart shows the time taken across timelines and resources. "
            f"Largest bubble: Timeline '{largest_bubble['Timelines']}', Resource '{largest_bubble['Resource']}' with {largest_bubble['Time_taken_hours']:.2f} hours. "
            f"Smallest bubble: Timeline '{smallest_bubble['Timelines']}', Resource '{smallest_bubble['Resource']}' with {smallest_bubble['Time_taken_hours']:.2f} hours. "
            f"Dominant activity type: '{dominant_activity}'."
        )

        
        summary_text = data_description

        return fig_bubble, summary_text
    except Exception as e:
        print(f"Error in Bubble Plot Callback: {e}")
        return {}, "Error: Unable to generate summary."

@app.callback(
    dash.dependencies.Output('multi-x-axis-columns', 'options'),
    [dash.dependencies.Input('x-axis-column', 'value'),
     dash.dependencies.Input('uploaded-file-path', 'data')]
)
def update_multi_x_axis_dropdown(x_axis_column, uploaded_file_path):
    file_path = uploaded_file_path
    if not file_path:
        return []
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Time_taken'], inplace=True)
        dropdown_options = df.columns.tolist()
        return [{'label': name, 'value': name} for name in dropdown_options]
    except Exception as e:
        print(f"Error in Multi X-Axis Dropdown Update Callback: {e}")
        return []

@app.callback(
    [dash.dependencies.Output('stacked-bar-chart', 'figure'),
     dash.dependencies.Output('top-activities-table', 'data')],
    [dash.dependencies.Input('uploaded-file-path', 'data'),
     dash.dependencies.Input('threshold-line1', 'value'),
     dash.dependencies.Input('threshold-line2', 'value')]
)
def update_stacked_bar_chart(uploaded_file_path, threshold1, threshold2):
    file_path = uploaded_file_path
    if not file_path:
        return {}, []
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Time_taken'], inplace=True)

        df['Time_taken_FTE'] = df['Time_taken'] / (60 * 8.50)

        grouped_df = df.groupby(['Timelines', 'Activity_type'])['Time_taken_FTE'].sum().reset_index()

        fig = px.bar(
            grouped_df,
            x='Timelines',
            y='Time_taken_FTE',
            color='Activity_type',
            barmode='stack',
            title='Work Load Distribution By Work Day (FTE)',
            labels={'Timelines': 'Work Day', 'Time_taken_FTE': 'Time Taken (FTE)'},
            color_continuous_scale='sunsetdark'
        )

        if threshold1 is not None:
            fig.add_shape(
                type='line',
                x0=grouped_df['Timelines'].min(),
                x1=grouped_df['Timelines'].max(),
                y0=threshold1,
                y1=threshold1,
                line=dict(color='red', dash='dot')
            )
            fig.add_annotation(
                xref='paper',
                yref='y',
                x=0.5,
                y=threshold1,
                text=f'{threshold1} FTE',
                showarrow=False,
                font=dict(color='red')
            )
        if threshold2 is not None:
            fig.add_shape(
                type='line',
                x0=grouped_df['Timelines'].min(),
                x1=grouped_df['Timelines'].max(),
                y0=threshold2,
                y1=threshold2,
                line=dict(color='orange', dash='dot')
            )
            fig.add_annotation(
                xref='paper',
                yref='y',
                x=0.5,
                y=threshold2,
                text=f'{threshold2} FTE',
                showarrow=False,
                font=dict(color='orange')
            )

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend_title_text='Activity Type',
            xaxis_title='Work Day',
            yaxis_title='Time Taken (FTE)',
            title_x=0.5,
            title_font=dict(size=20, family='Calibri', color='black'),
            font=dict(family='Calibri', size=12, color='black'),
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(tickfont=dict(family='Calibri', size=12, color='black')),
            yaxis=dict(tickfont=dict(family='Calibri', size=12, color='black'))
        )

        fig.update_yaxes(tickformat=".2f")
        
        top_activities = (
            df.groupby(['Timelines', 'Activity_type'])['Time_taken_FTE']
            .sum()
            .reset_index()
            .sort_values(by=['Timelines', 'Time_taken_FTE'], ascending=[True, False])
        )
        top_activities = (
            top_activities.groupby('Timelines')
            .head(3)
            .reset_index(drop=True)
        )
        top_activities['Time_taken_FTE'] = top_activities['Time_taken_FTE'].round(2)
        table_data = top_activities.to_dict('records')

        return fig, table_data

    except Exception as e:
        print(f"Error in Stacked Bar Chart Callback: {e}")
        return {}, []

@app.callback(
    dash.dependencies.Output('prediction-data-download', 'data'),
    [dash.dependencies.Input('export-button-pred', 'n_clicks')],
    [dash.dependencies.State('prediction-data-store', 'data')],
    prevent_initial_call=True
)
def export_level_loading(n_clicks, prediction_data):
    if n_clicks > 0 and prediction_data:
        df = pd.DataFrame(prediction_data)

        temp_dir = os.path.join(base_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, "level_loading_recommendations.csv")
        df.to_csv(temp_file_path, index=False)
        temp_files.append(temp_file_path)
        return dcc.send_file(temp_file_path)
    return None

@app.callback(
    dash.dependencies.Output('input-sheet-download', 'data'),
    [dash.dependencies.Input('input-sheet', 'n_clicks')],
    prevent_initial_call=True
)
def export_input_sheet(n_clicks):
    if n_clicks > 0:

        input_template = pd.DataFrame(columns=[
            'Timelines', 'Activity_type', 'Task_name', 'Time_taken', 'Resource', 'Entity'
        ])

        temp_dir = os.path.join(base_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, "input_sheet_template.csv")
        input_template.to_csv(temp_file_path, index=False)
        temp_files.append(temp_file_path)
        return dcc.send_file(temp_file_path)
    return None

@app.callback(
    dash.dependencies.Output('exit-button', 'n_clicks'),
    [dash.dependencies.Input('exit-button', 'n_clicks')]
)
def handle_exit(n_clicks):
    if n_clicks > 0:
        exit_app()
    return n_clicks

@app.callback(
    dash.dependencies.Output('multi-data-download', 'data'),
    [dash.dependencies.Input('export-multi-button', 'n_clicks')],
    [dash.dependencies.State('uploaded-file-path', 'data')],
    prevent_initial_call=True
)
def export_multi_recommendations(n_clicks, uploaded_file_path):
    if n_clicks <= 0 or not uploaded_file_path:
        return None

    try:
        df = pd.read_csv(uploaded_file_path)
        df.dropna(subset=['Time_taken'], inplace=True)
        df['Time_taken_FTE'] = df['Time_taken'] / (60 * 8.50)

        top_days = df.groupby('Timelines')['Time_taken_FTE'].sum().sort_values(ascending=False).head(3).index.tolist()

        all_recommendations = []

        for day in top_days:
            daily_df = df[df['Timelines'] == day]

            top_activities = (
                daily_df.groupby('Activity_type')['Time_taken_FTE']
                .sum()
                .sort_values(ascending=False)
                .head(3)
                .index
                .tolist()
            )

            for activity in top_activities:
                sub_df = daily_df[daily_df['Activity_type'] == activity].copy()
                result_df = perform_clustering(sub_df, quantile_value=75)
                result_df['Selected_Timelines'] = day
                result_df['Selected_Activity_type'] = activity
                all_recommendations.append(result_df)

        final_df = pd.concat(all_recommendations, ignore_index=True)

        temp_dir = os.path.join(base_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, "multi_day_activity_recommendations.csv")
        final_df.to_csv(temp_file_path, index=False)
        temp_files.append(temp_file_path)

        return dcc.send_file(temp_file_path)

    except Exception as e:
        print(f"Error in multi-export: {e}")
        return None


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    _, port = s.getsockname()
    s.close()
    return port

if __name__ == '__main__':
    port = find_free_port()
    def open_browser():
        webbrowser.open_new_tab(f'http://127.0.0.1:{port}/')
    Timer(1, open_browser).start()
    app.run(port=port)
