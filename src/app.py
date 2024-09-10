import dash
from dash import dcc, html, Input, Output, callback, State, dash_table, Dash
import dash_bootstrap_components as dbc




import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_validate ,ParameterGrid, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC

from sklearn.utils import shuffle


df = pd.read_csv('training_data.csv')

models = {"knn": KNeighborsClassifier(), "LogReg": LogisticRegression(random_state=5),
          "DecTree": DecisionTreeClassifier(random_state=5), "RandForest": RandomForestClassifier(random_state=5),
          "HGB":HistGradientBoostingClassifier(random_state=5), "SVC":SVC(random_state=5)}

grids = {"knn":{'n_neighbors': range(1,11), 'weights':['uniform','distance']},
         "LogReg": {'solver':['lbfgs','newton-cg'], "C":[0.01, 0.1, 1, 10, 100]}, 
         "DecTree":{"criterion":["gini","entropy"], "max_depth":[5,10,None]}, 
         "RandForest": {"criterion":["gini","entropy"], "max_depth":[5,10,None]},
         "HGB": {"max_depth": [5,10,None], "learning_rate": [0.1, 0.5, 1]},
         "SVC": {"kernel": ["linear", "poly"], "C": [0.01, 0.1, 1, 10, 100]}}

scalers = {"No scaling":0,'StandardScaler': StandardScaler(), "MinMaxScaler": MinMaxScaler()}

scores = []
scores_dtc = []
hover_data_ini = {"Scaler":[], 'Weights':[], 'n_neighbors':[], 'p_value':[]}
hover_data_ini_dtc = {"Scaler":[], 'Criterion':[], 'max_depth':[], 'min_sample':[]}
# Extracting features and targets
y = df['Class'].values
X = df.drop('Class',axis=1).values


# Create a Dash web application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Define the layout of the app
app.layout = html.Div(style={'margin':'15px'},
    children=[
        html.H2("Machine Learning Task", style={"text-align":"center"}),
        html.H3("Input Data Exploration"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(
                    [
                        html.H4("Parameters", style={"text-align":"center"}),
                        html.P(
                            "Select the parameters for visualization.",
                            className="card-text",
                        ),
                        html.P("Feature on X axis"),
                        dcc.Dropdown(df.columns,
                                     "Red",id="exp_x",
                                     style={'width': '100%','background-color':'#a2b4c6','color':'#222222'},clearable=False),
                        html.P("Feature on Y axis"),
                        dcc.Dropdown(df.columns,
                                     "Green",id="exp_y",
                                     style={'width': '100%','background-color':'#a2b4c6','color':'#222222'},clearable=False),
                        html.P("Feature on Z axis"),
                        dcc.Dropdown(df.columns,
                                     "Blue",id="exp_z",
                                     style={'width': '100%','background-color':'#a2b4c6','color':'#222222'},clearable=False),
                        html.P("scaling method"),
                        dcc.Dropdown(list(scalers.keys()),
                                     "No scaling",id="scaler",
                                     style={'width': '100%','background-color':'#a2b4c6','color':'#222222'},clearable=False),
                        html.P("Data Range"),
                        dcc.RangeSlider(min=0, max=48142, step=1, value=[0,1000], id='drange', marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True}),
                        
                        
            ])
        ]),

        ],width=3),
            dbc.Col([
                dcc.Graph(figure={},id='exp2_plot'),
                
                              
                
                
            ],width=5),
            dbc.Col([
                dcc.Graph(figure={},id='exp3_plot'),
                
                              
                
                
            ],width=4),

    ]),
        html.Hr(),
        html.H3("Preliminary model exploration. K-Nearest neighbors"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(
                    [
                        html.H4("HyperParameters for knn ", style={"text-align":"center"}),
                        html.P("Scaler"),
                        dcc.Dropdown(list(scalers.keys()),
                                     "No scaling",id="knn_scaler",
                                     style={'width': '100%','background-color':'#a2b4c6','color':'#222222'},clearable=False),
                        html.P("Weights"),
                        dcc.Dropdown(['uniform','distance'],
                                     "uniform",id="knn_weights",
                                     style={'width': '100%','background-color':'#a2b4c6','color':'#222222'},clearable=False),
                        
                        html.P("n_neighbors"),
                        dcc.Slider(min=1, max=50, step=1, value=5, id='knn_n_neighbors', marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.P("p value"),
                        dcc.Slider(min=0.1, max=10, step=0.1, value=2, id='p_val', marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True}),
                        html.P("Leaf Size"),
                        dcc.Slider(min=1, max=100, step=1, value=30, id='leaf_size', marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True}),
                        
                        
            ])
        ]),

        ],width=3),
            dbc.Col([
                dcc.Graph(figure={},id='knn_ini'),
                
                              
                
                
            ],width=3),
            dbc.Col([
                dcc.Graph(figure={},id='knn_ini2'),
                
                              
                
                
            ],width=2),
            dbc.Col([
                dcc.Graph(figure={},id='knn_ini3'),
                
                              
                
                
            ],width=4),

    ]),
        html.Hr(),
        html.H3("Preliminary model exploration. Decision Tree"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(
                    [
                        html.H4("HyperParameters for DecTree ", style={"text-align":"center"}),
                        html.P("Scaler"),
                        dcc.Dropdown(list(scalers.keys()),
                                     "No scaling",id="DecTree_scaler",
                                     style={'width': '100%','background-color':'#a2b4c6','color':'#222222'},clearable=False),
                        html.P("Criterion"),
                        dcc.Dropdown(["gini", "entropy", "log_loss"],
                                     "gini",id="DecTree_crit",
                                     style={'width': '100%','background-color':'#a2b4c6','color':'#222222'},clearable=False),
                        
                        html.P("max_depth"),
                        dcc.Slider(min=1, max=25, step=1, value=5, id='DecTree_max_depth', marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.P("min_samples_split"),
                        dcc.Slider(min=1, max=10, step=1, value=2, id='DecTree_min_sample', marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True}),
                        
                        
                        
            ])
        ]),

        ],width=3),
            dbc.Col([
                dcc.Graph(figure={},id='DecTree_ini1'),
                
                              
                
                
            ],width=3),
            dbc.Col([
                dcc.Graph(figure={},id='DecTree_ini2'),
                
                              
                
                
            ],width=2),
            dbc.Col([
                dcc.Graph(figure={},id='DecTree_ini3'),
                
                              
                
                
            ],width=4),

    ]),
        
        
    ])

#Update for input data exploration
@callback(
    Output(component_id='exp2_plot', component_property='figure'),
    Input(component_id='exp_x', component_property='value'),
    Input(component_id='exp_y', component_property='value'),
    Input(component_id='exp_z', component_property='value'),
    Input(component_id='scaler', component_property='value'),
    Input(component_id='drange', component_property='value'),    
)

def update_expl(exp_x,exp_y,exp_z,sc,drange):
    dfh = shuffle(df,random_state=5)
    dfh = dfh.iloc[drange[0]:drange[1]]
    dfh["class"] = df["Class"].astype(str)
    
    fig = px.scatter(dfh,x=exp_x,y=exp_y,color="class",opacity=0.7)
    fig.update_layout(template='plotly_dark')
    fig.layout.xaxis.title.text= exp_x
    fig.layout.yaxis.title.text= exp_y
    fig.layout.title.text = "Data points"
    fig.update_traces(marker={'size': 6})
    return fig
    
    
@callback(
    Output(component_id='exp3_plot', component_property='figure'),
    Input(component_id='exp_x', component_property='value'),
    Input(component_id='exp_y', component_property='value'),
    Input(component_id='exp_z', component_property='value'),
    Input(component_id='scaler', component_property='value'),
    Input(component_id='drange', component_property='value'),    
)
def update_expl2(exp_x,exp_y,exp_z,sc,drange):
    dfh = shuffle(df,random_state=5)
    dfh = dfh.iloc[drange[0]:drange[1]]
    dfh["class"] = df["Class"].astype(str)
    
    fig = px.scatter_3d(dfh,x=exp_x,y=exp_y,z=exp_z,color="class", opacity=0.7)
    fig.update_layout(template='plotly_dark')
    fig.layout.xaxis.title.text= exp_x
    fig.layout.yaxis.title.text= exp_y
    fig.layout.title.text = "Data points 3D"
    fig.update_traces(marker={'size': 4})
    return fig

@callback(
    Output(component_id='knn_ini', component_property='figure'), 
    Input(component_id='knn_scaler', component_property='value'),       
    Input(component_id='knn_weights', component_property='value'),
    Input(component_id='knn_n_neighbors', component_property='value'),
    Input(component_id='p_val', component_property='value'),            
)
def update_ini1(sc,weight,n,p):   
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=5, stratify=y)
    
    if sc=='No scaling':
        pass
    else:
        scaler = scalers[sc]
        X_train =scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    knn = KNeighborsClassifier(n_neighbors=n, weights=weight, p=p)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    np.bool = np.bool_    
    
    fig = px.imshow(cm, text_auto=True)   
    fig.update_layout(template='plotly_dark')
    fig.layout.xaxis.title.text= "true label"
    fig.layout.yaxis.title.text= "predicted label"
    fig.layout.title.text = "Confusion matrix for knn" 
    fig.update_yaxes(tick0=0, dtick=1)    
     
    return fig

@callback(
    Output(component_id='knn_ini2', component_property='figure'), 
    Input(component_id='knn_scaler', component_property='value'),       
    Input(component_id='knn_weights', component_property='value'),
    Input(component_id='knn_n_neighbors', component_property='value'),
    Input(component_id='p_val', component_property='value'),
            
)
def update_ini2(sc,weight,n,p):   
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=5, stratify=y)
    
    if sc=='No scaling':
        pass
    else:
        scaler = scalers[sc]
        X_train =scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    knn = KNeighborsClassifier(n_neighbors=n, weights=weight, p=p)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1score = f1_score(y_test, y_pred, average='macro')
        
    
    fig = px.bar(x=['accuracy', 'precision', 'recall', 'f1 score'],y=[1-accuracy,1-precision,1-recall,1-f1score],)   
    fig.update_layout(template='plotly_dark')
    fig.layout.xaxis.title.text= "metrics"
    fig.layout.yaxis.title.text= "1 - score values"
    fig.layout.yaxis.range=[0,0.01]
    fig.layout.title.text = "Metric scores lower score better"       
     
    return fig

@callback(
    Output(component_id='knn_ini3', component_property='figure'), 
    Input(component_id='knn_scaler', component_property='value'),       
    Input(component_id='knn_weights', component_property='value'),
    Input(component_id='knn_n_neighbors', component_property='value'),
    Input(component_id='p_val', component_property='value'),
            
)

def update_ini3(sc,weight,n,p):   
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=5, stratify=y)
    
    if sc=='No scaling':
        pass
    else:
        scaler = scalers[sc]
        X_train =scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    knn = KNeighborsClassifier(n_neighbors=n, weights=weight, p=p)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1score = f1_score(y_test, y_pred, average='macro')
    scores.append([accuracy,precision,recall,f1score])
    hover_data_ini['Scaler'].append(sc)
    hover_data_ini['Weights'].append(weight)
    hover_data_ini['n_neighbors'].append(n)
    hover_data_ini['p_value'].append(p)    
    hover_df = pd.DataFrame(hover_data_ini)    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.array(scores)[:,0], mode='lines+markers', name='accuracy',hoverinfo='none'))
    fig.add_trace(go.Scatter(y=np.array(scores)[:,1], mode='lines+markers', name='precision',hoverinfo='none'))
    fig.add_trace(go.Scatter(y=np.array(scores)[:,2], mode='lines+markers', name='recall',hoverinfo='none'))
    fig.add_trace(go.Scatter(y=np.array(scores)[:,3], mode='lines+markers', name='f1-score',customdata = hover_df[['Scaler','Weights','n_neighbors']],
                  hovertemplate='Scaler:%{customdata[0]} <br>Weights:%{customdata[1]}<br>n_neighbors:%{customdata[2]}'))
    fig.update_layout(template='plotly_dark')
    fig.update_layout(hovermode='x')
    fig.layout.xaxis.title.text= "setting_no"
    fig.layout.yaxis.title.text= "score values"    
    fig.layout.title.text = "Different metric scores for selected params, lower the score, better"       
     
    return fig

@callback(
    Output(component_id='DecTree_ini1', component_property='figure'), 
    Input(component_id='DecTree_scaler', component_property='value'),       
    Input(component_id='DecTree_crit', component_property='value'),
    Input(component_id='DecTree_max_depth', component_property='value'),
    Input(component_id='DecTree_min_sample', component_property='value'),            
)
def update_Dini1(sc,crit,md,ms):   
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=5, stratify=y)
    
    if sc=='No scaling':
        pass
    else:
        scaler = scalers[sc]
        X_train =scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    dtc = DecisionTreeClassifier(criterion=crit, max_depth=md, min_samples_split=ms, random_state=5)
    dtc.fit(X_train,y_train)
    y_pred = dtc.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    np.bool = np.bool_    
    
    fig = px.imshow(cm, text_auto=True)   
    fig.update_layout(template='plotly_dark')
    fig.layout.xaxis.title.text= "true label"
    fig.layout.yaxis.title.text= "predicted label"
    fig.layout.title.text = "Confusion matrix for knn" 
    fig.update_yaxes(tick0=0, dtick=1)    
     
    return fig

@callback(
    Output(component_id='DecTree_ini2', component_property='figure'), 
    Input(component_id='DecTree_scaler', component_property='value'),       
    Input(component_id='DecTree_crit', component_property='value'),
    Input(component_id='DecTree_max_depth', component_property='value'),
    Input(component_id='DecTree_min_sample', component_property='value'),            
)
def update_Dini2(sc,crit,md,ms):     
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=5, stratify=y)
    
    if sc=='No scaling':
        pass
    else:
        scaler = scalers[sc]
        X_train =scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    dtc = DecisionTreeClassifier(criterion=crit, max_depth=md, min_samples_split=ms, random_state=5)
    dtc.fit(X_train,y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1score = f1_score(y_test, y_pred, average='macro')
        
    
    fig = px.bar(x=['accuracy', 'precision', 'recall', 'f1 score'],y=[1-accuracy,1-precision,1-recall,1-f1score],)   
    fig.update_layout(template='plotly_dark')
    fig.layout.xaxis.title.text= "metrics"
    fig.layout.yaxis.title.text= "1 - score values"
    fig.layout.yaxis.range=[0,0.01]
    fig.layout.title.text = "Metric scores lower score better"       
     
    return fig

@callback(
    Output(component_id='DecTree_ini3', component_property='figure'), 
    Input(component_id='DecTree_scaler', component_property='value'),       
    Input(component_id='DecTree_crit', component_property='value'),
    Input(component_id='DecTree_max_depth', component_property='value'),
    Input(component_id='DecTree_min_sample', component_property='value'),            
)
def update_Dini3(sc,crit,md,ms):     
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=5, stratify=y)
    
    if sc=='No scaling':
        pass
    else:
        scaler = scalers[sc]
        X_train =scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    dtc = DecisionTreeClassifier(criterion=crit, max_depth=md, min_samples_split=ms, random_state=5)
    dtc.fit(X_train,y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1score = f1_score(y_test, y_pred, average='macro')
    scores_dtc.append([accuracy,precision,recall,f1score])
    hover_data_ini_dtc['Scaler'].append(sc)
    hover_data_ini_dtc['Criterion'].append(crit)
    hover_data_ini_dtc['max_depth'].append(md)
    hover_data_ini_dtc['min_sample'].append(ms)    
    hover_df = pd.DataFrame(hover_data_ini_dtc)    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.array(scores_dtc)[:,0], mode='lines+markers', name='accuracy',hoverinfo='none'))
    fig.add_trace(go.Scatter(y=np.array(scores_dtc)[:,1], mode='lines+markers', name='precision',hoverinfo='none'))
    fig.add_trace(go.Scatter(y=np.array(scores_dtc)[:,2], mode='lines+markers', name='recall',hoverinfo='none'))
    fig.add_trace(go.Scatter(y=np.array(scores_dtc)[:,3], mode='lines+markers', name='f1-score',customdata = hover_df,
                  hovertemplate='Scaler:%{customdata[0]} <br>Criterion:%{customdata[1]}<br>max_depth:%{customdata[2]}<br>min_sample_split:%{customdata[]}'))
    fig.update_layout(template='plotly_dark')
    fig.update_layout(hovermode='x')
    fig.layout.xaxis.title.text= "setting_no"
    fig.layout.yaxis.title.text= "score values"    
    fig.layout.title.text = "Different metric scores for selected params, lower the score, better"       
     
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
 