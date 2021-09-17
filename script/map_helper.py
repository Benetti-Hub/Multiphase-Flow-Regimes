import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from script import utils

di = {0: 'Annular', 1:'Dispersed', 
      2:'Intermittent', 3:'Stratified', 
      4:'Stratified', 5:'Dispersed'}

def bar_plot_studies(df_test, model, kept_columns,  n_class=6):
    
    '''
    Another simple function to show the accuracy on the single studies
    '''
    
    author_list = list(df_test['Author'].value_counts().index)
    model_info = pd.DataFrame(index=author_list, columns=["Model"])

    for author in author_list:

        di = {}

        df_author = df_test.loc[(df_test['Author']==author)]
        df_author = utils.bronze_to_gold(df_author, kept_columns=kept_columns+['Flow_label'])
        X_test = df_author.iloc[:,:-1].values
        y_test = df_author.iloc[:,[-1]].values.ravel()

        
        y_pred = model.predict(X_test)
        y_pred = y_pred if y_pred.ndim==1 else y_pred.argmax(1)
        if n_class==4:
            y_pred = np.where(y_pred==5, 1, y_pred)
            y_pred = np.where(y_pred==4, 3, y_pred)
            
        model_info.loc[author] = accuracy_score(y_test, y_pred)

    model_info = model_info.sort_index()
    fig = go.Figure(go.Bar(
        y=model_info['Model'], x=model_info.index))
    fig.add_hline(y=model_info['Model'].mean(), line_dash="dash", line_color="red")
    fig.update_yaxes(title="Accuracy")
    fig.update_xaxes(title="Independent Study", tickangle=45)
    fig.update_layout(barmode='group', title="Prediction Accuracy on different studies")
    
    fig.show()
    
    return model_info



C_cs=[
    [0.0, 'rgba(128,0,0,255)'],
    [1.0, 'rgba(200,200,255,255)'],
]

Z_cs=[
    [0.0, 'rgba(0,0,0,0)'],
    [0.8, 'rgba(0,0,0,0)'],
    [0.8, 'rgba(0,0,0,255)'],
    [1.0, 'rgba(0,0,0,255)'],
]

cs_di = {'Z_img': Z_cs, 
         'conf': 'RdBu', 
         'Flow_label': 'RdBu'}

def get_heatmap_data(model, columns, di_val, di_v, mesh_size=100):
    
    '''
    Return a mesh of fixed experimental quantities with the
    prediction of the model
    '''
    
    map_x, map_y = np.meshgrid(np.logspace(di_v['VsL_min'], 
                                           di_v['VsL_max'], 
                                           num=mesh_size),
                               np.logspace(di_v['VsG_min'], 
                                           di_v['VsG_max'], 
                                           num=mesh_size))
    
    mapping = np.array([map_x, map_y]).reshape(2, -1).T

    df_mesh = pd.DataFrame(mapping, columns=['Vsl','Vsg']) 
    df_mesh['DenL'] = 1000
    df_mesh['DenG'] = 1.12
    df_mesh['VisL'] = 0.001
    df_mesh['VisG'] = 0.000018 
    df_mesh['ST']   = 0.07
    df_mesh['ID']   = 0.051
    df_mesh['Ang']  = 0
    
    for k, v in di_val.items(): 
        df_mesh[k] = v
            
    X_to_predict = utils.generate_features(df_mesh)[columns].values
    
    regime =  model.predict(X_to_predict)
    if regime.ndim==1:
        proba = model.predict_proba(X_to_predict)
        proba[:, 1] = proba[:, 1] + proba[:, 5]
        proba[:, 3] = proba[:, 3] + proba[:, 4]
        proba = proba[:,:4]
        
    else:
        proba = regime
        proba[:, 1] = proba[:, 1] + proba[:, 5]
        proba[:, 3] = proba[:, 3] + proba[:, 4]
        proba = proba[:,:4]
        
        regime = regime.argmax(1)
        
    regime = np.where(regime==4, 3, regime)
    regime = np.where(regime==5, 1, regime)
    
    df_mesh = pd.DataFrame(mapping, columns=['Vsl','Vsg'])
    df_mesh['log(Vsl)'] = np.log10(df_mesh['Vsl'])
    df_mesh['log(Vsg)'] = np.log10(df_mesh['Vsg'])
    df_mesh['Flow_label'] = regime
    df_mesh["Flow Regime"] = df_mesh['Flow_label'].replace(di)
    df_mesh['conf'] = proba.max(axis=-1) - (proba.sum(axis=-1) - proba.max(axis=-1))
    
    return df_mesh

def write_map_images(df_map, model, columns):
    
    auth = list(df_map['Author'].value_counts().index)
    for Author in auth:
        df_test = df_map.loc[df_map['Author']==Author]
        maps = list(df_test['Fig'].value_counts().index)

        for paper_figure in maps:

            df_paper = df_test.loc[df_test['Fig']==paper_figure].copy()

            di_v = {'VsL_max' : np.log10(1.5*df_paper['Vsl'].max()),
                    'VsL_min' : np.log10(0.5*df_paper['Vsl'].min()),
                    'VsG_max' : np.log10(1.5*df_paper['Vsg'].max()),
                    'VsG_min' : np.log10(0.5*df_paper['Vsg'].min())}


            df_mesh = get_heatmap_data(model, columns, df_paper.min().iloc[:-3].to_dict(), di_v)

            X_test = utils.bronze_to_gold(df_paper)[columns].values

            y_pred = model.predict(X_test)
            y_pred = y_pred if y_pred.ndim==1 else y_pred.argmax(1)
            y_pred = np.where(y_pred==5, 1, y_pred)
            y_pred = np.where(y_pred==4, 3, y_pred)

            df_paper["Predicted"] = y_pred
            df_paper["Correct"] = (df_paper['Flow_label']==df_paper['Predicted'].replace(di))
            df_paper = df_paper.sort_values(by='Correct', ascending=False)

            fig = px.scatter(
                df_paper, 
                x='Vsg', y='Vsl', symbol='Flow_label', color='Correct',
                color_discrete_sequence=["green", "red"],
                symbol_map = {'Annular' : 'square-dot',
                              'Dispersed' : 'circle-dot',
                              'Intermittent' : 'x-dot',
                              'Stratified' : 'triangle-left-dot'              
                             },
            )

            fig.update_traces(
                marker_size=12, marker_line_width=1.5,
            )

            fig.add_trace(
                go.Heatmap( 
                    x=df_mesh['Vsg'],
                    y=df_mesh['Vsl'],
                    z=df_mesh['Flow_label'],
                    zmin=0, zmax=3,

                    colorbar=dict(
                        tickmode="array",
                        tickvals=[0, 1, 2, 3],
                        ticktext=["Annular", "Dispersed", "Intermittent", "Stratified"],
                        ticks="outside"
                    ),

                    colorscale='RdBu',
                    opacity=0.8,
                )
            )

            fig.update_layout(
                legend_orientation='h',
                width=800,
                height=600,
                title=f'Prediction on {Author} Data: figure {paper_figure}'
            )

            fig.update_xaxes(title=r"$Us_{G} (m/s)$", type="log")
            fig.update_yaxes(title=r"$Us_{L} (m/s)$", type="log")

            print(f'{Author} figure: {paper_figure}')
            di_map = df_paper[['Author', 'Fig', 'ID', 
                               'Ang', 'Type of liquid', 
                               'Type of Gas']].iloc[3].to_dict()
            
            st = str()
            for key, val in di_map.items():
                if isinstance(val, float):
                    val = '%s' % float('%.3g' % val)

                st = st+f"{key}_{val}_".replace(" ", "")

            fig.write_image("Plots/MultiphaseMaps/"
                            f"Secret/Predicted_{len(columns)}/{st}.png", 
                            scale=2) 
            
def probability_threshold(model, X_secret, y_secret):
    
    y_proba = model.predict_proba(X_secret)

    y_proba[:, 1] = y_proba[:, 1] + y_proba[:, 5]
    y_proba[:, 3] = y_proba[:, 3] + y_proba[:, 4]
    y_proba = y_proba[:,:4]

    df_predicted = pd.DataFrame(y_proba, columns=["Annular", "Dispersed", "Intermittent", "Stratified"])
    df_predicted["Pmax"] = df_predicted.max(axis=1)
    df_predicted["Predicted"] = y_proba.argmax(1)
    df_predicted["GT"] = y_secret

    p_max = np.linspace(0.8, 0.9997, 20)

    infos = np.zeros((len(p_max), 3))
    for i, prob in enumerate(p_max):
        df_tomax = df_predicted.loc[df_predicted["Pmax"]>prob][['Predicted', 'GT']]
        acc = accuracy_score(df_tomax['Predicted'], df_tomax['GT'])

        infos[i, :] = acc, df_tomax.shape[0]/df_predicted.shape[0], prob
        
    info_df = pd.DataFrame(infos, columns=['Accuracy', 'Data', 'Prob'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title_text="Probability Threshold Effects on Accuracy"
    )

    fig.add_trace(go.Bar(x=info_df["Prob"], y=info_df["Data"], name="Data"), secondary_y=False)
    fig.add_trace(go.Scatter(x=info_df["Prob"], y=info_df['Accuracy'], name="Accuracy"), secondary_y=True)

    fig.update_yaxes(title_text="Fraction of original data", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
    fig.update_xaxes(title_text="Probability Threshold")
    
    fig.show()
    
    return df_predicted
    
def get_top_two(df_predicted):
    
    def get_2_largest(df):
        if isinstance(df, pd.DataFrame):
            return df.aggregate(get_2_largest)
        else:
            return df.nlargest(2)
    
    largest = get_2_largest(df_predicted[["Annular", "Dispersed", "Intermittent", "Stratified"]].T).T
    low = largest.idxmin(axis=1)
    di_t = {'Annular': 0, 'Dispersed':1, 'Intermittent':2, 'Stratified':3} 
    df_predicted["Second"] = low.replace(di_t)
    df_predicted['acc_sec'] = [(df_predicted['GT']==df_predicted['Predicted']) | (df_predicted['GT']==df_predicted['Second'])][0]
    
    print(f"Accuracy of top 2: {round(df_predicted['acc_sec'].sum()/df_predicted.shape[0], 4)}")

def load_image(path, filtering=200, size=100):
    
    img = Image.open(path)
    img_res = ImageOps.flip(img.resize((size,size)))
    img_arr = np.array(img_res)
    
    black_pixels_mask = np.all(img_arr<=[filtering, filtering, filtering, 255], axis=-1)
    img_arr[black_pixels_mask]  = [0,0,0,1]
    img_arr[~black_pixels_mask] = [0,0,0,0]
    
    return img_arr
    
    
def plot_madhane_map(model, columns, mesh_size=400):
    
    di = {0: 'A', 1: 'DB', 2: 'I', 3: 'SW', 4: 'SS', 5:'B'}
    
    map_x, map_y = np.meshgrid(np.logspace(-2, 1.301, num=mesh_size), np.logspace(-1, 2.7, num=mesh_size))
    mapping = np.array([map_x, map_y]).reshape(2, -1).T

    df_mesh = pd.DataFrame(mapping, columns=['Vsl','Vsg'])
    
    df_mesh['Vsl'] = df_mesh['Vsl']/3.28084
    df_mesh['Vsg'] = df_mesh['Vsg']/3.28084
    df_mesh['DenL'] = 1000
    df_mesh['DenG'] = 1.12
    df_mesh['VisL'] = 0.001
    df_mesh['VisG'] = 0.000018 
    df_mesh['ST']   = 0.07
    df_mesh['ID']   = 0.051
    df_mesh['Ang']  = 0
    Z = load_image('Plots/MultiphaseMaps/Madhane_2.png', 
                   size=mesh_size)[:,:,3].T.reshape(-1,1)
        
    X_to_predict = utils.generate_features(df_mesh)[columns].values
    
    regime =  model.predict(X_to_predict)
    if regime.ndim==1:
        proba = model.predict_proba(X_to_predict)
    else:
        proba = regime
        regime = regime.argmax(1)
    
    df_mesh = pd.DataFrame(mapping, columns=['Vsl','Vsg'])
    
    df_mesh['Vsl'] = df_mesh['Vsl']*3.28084
    df_mesh['Vsg'] = df_mesh['Vsg']*3.28084
    df_mesh['log(Vsl)'] = np.log10(df_mesh['Vsl'])
    df_mesh['log(Vsg)'] = np.log10(df_mesh['Vsg'])
    df_mesh['Flow_label'] = regime
    df_mesh["Flow Regime"] = df_mesh['Flow_label'].replace(di)
    df_mesh['conf'] = proba.max(axis=-1) - (proba.sum(axis=-1) - proba.max(axis=-1))
    df_mesh['Z_img']=Z    
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap( 
            x=df_mesh['Vsg'],
            y=df_mesh['Vsl'],
            z=df_mesh['Flow_label'],
            zmin=0, zmax=5,
            colorscale='RdBu',
            opacity=0.8,
            customdata=df_mesh['Flow Regime'],
            hovertemplate=(
                'Flow Regime: %{customdata} <br><extra></extra>'
                'Confidence: %{z:.3f} <br>'

            ),
            colorbar=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=["Annular", "Dispersed", "Intermittent",
                          "Stratified Wavy", "Stratified Smooth", "Bubbly"],
                
                ticks="outside"
            ),
        )
    )

    fig.add_trace(
        go.Heatmap( 
            x=df_mesh['Vsg'],
            y=df_mesh['Vsl'],
            z=df_mesh['Z_img'],
            zmin=0, zmax=1,
            colorscale=cs_di['Z_img'],
            opacity=1,
            customdata=df_mesh['Flow Regime'],
            hovertemplate=(
                'Flow Regime: %{customdata} <br><extra></extra>'
                'Confidence: %{z:.3f} <br>'
            ), showscale=False
        )
    )
    fig.update_xaxes(type="log", title=r"$Us_{G} (ft/s)$")
    fig.update_yaxes(type="log", title=r"$Us_{L} (ft/s)$")

    fig.update_layout(
        legend_orientation='h',
        width=800,
        height=600,
        title=f"Comparison with Madhane's Map",
    )

    fig.show()
    
def test_set_comparison(model, columns, df_test, display="conf"):
        
    di = {0: 'A', 1: 'DB', 2: 'I', 3: 'SW', 4: 'SS', 5:'B'}
    df_mesh = get_heatmap_data_6(model, columns, df_test.drop(columns=['Vsl', 'Vsg']).min().to_dict())
    
    X_test = utils.bronze_to_gold(df_test)[columns].values
    y_pred = model.predict(X_test)
    
    df_test["Predicted"] = y_pred
    df_test["Correct"] = (df_test['Flow_label']==df_test['Predicted'].replace(di))
    df_test = df_test.sort_values(by='Correct', ascending=False)
   
    fig = px.scatter(
        df_test, 
        x='Vsg', y='Vsl', symbol='Flow_label', color='Correct',
        color_discrete_sequence=["green", "red"],
        symbol_map = {'A': 'square-dot', 
                      'DB': 'circle-dot', 
                      'I': 'x-dot',
                      'SW': 'triangle-left-dot',
                      'SS': 'hexagon-dot',
                      'B': 'star-triangle-up-dot'             
                     }
    )

    fig.update_traces(
        marker_size=12, marker_line_width=1.5,
    )
    
    if display=="Flow_label":
        zmax=5
        cbar = dict(tickmode="array",
                    tickvals=[0, 1, 2, 3, 4, 5],
                    ticktext=["Annular", "Dispersed", "Intermittent",
                              "Stratified Wavy", "Stratified Smooth", "Bubbly"],
                    ticks="outside"
            )
    else:
        zmax=1
        cbar=dict() 
    
    fig.add_trace(
        go.Heatmap( 
            x=df_mesh['Vsg'],
            y=df_mesh['Vsl'],
            z=df_mesh[display],
            zmin=0, zmax=zmax,
            colorscale='RdBu',
            opacity=0.8,
            customdata=df_mesh['Flow Regime'],
            hovertemplate=(
                'Flow Regime: %{customdata} <br><extra></extra>'
                'Confidence: %{z:.3f} <br>'

            ),
            colorbar=cbar,
        )
    )

    fig.update_xaxes(type="log", title=r"$Us_{G} (m/s)$")
    fig.update_yaxes(type="log", title=r"$Us_{L} (m/s)$")

    fig.update_layout(
        legend_orientation='h',
        width=800,
        height=600,
        title=f"Comparison with Test Data",
    )

    fig.show()
    
    
def get_heatmap_data_6(model, columns, di_val, mesh_size=100):

    map_x, map_y = np.meshgrid(np.logspace(-3.1, 1.1, num=mesh_size), np.logspace(-2.1, 2.1, num=mesh_size))
    mapping = np.array([map_x, map_y]).reshape(2, -1).T

    df_mesh = pd.DataFrame(mapping, columns=['Vsl','Vsg'])     
    df_mesh['DenL'] = 1000
    df_mesh['DenG'] = 1.12
    df_mesh['VisL'] = 0.001
    df_mesh['VisG'] = 0.000018 
    df_mesh['ST']   = 0.07
    df_mesh['ID']   = 0.051
    df_mesh['Ang']  = 0
    
    for k, v in di_val.items(): 
        df_mesh[k] = v
        if k == "Ang":
            Ang = int(df_mesh['Ang'].values[0])
    
    try:
        Z = load_image(f'Plots/MultiphaseMaps/BarneaMaps_{Ang}.png', 
                       size=mesh_size)[:,:,3].T.reshape(-1,1)
    except:
        Z = 0
        
    X_to_predict = utils.generate_features(df_mesh)[columns].values
    
    regime =  model.predict(X_to_predict)
    if regime.ndim==1:
        proba = model.predict_proba(X_to_predict)
    else:
        proba = regime
        regime = regime.argmax(1)
    
    df_mesh = pd.DataFrame(mapping, columns=['Vsl','Vsg'])    
    df_mesh['log(Vsl)'] = np.log10(df_mesh['Vsl'])
    df_mesh['log(Vsg)'] = np.log10(df_mesh['Vsg'])
    df_mesh['Flow_label'] = regime
    df_mesh["Flow Regime"] = df_mesh['Flow_label'].replace(di)
    df_mesh['conf'] = proba.max(axis=-1) - (proba.sum(axis=-1) - proba.max(axis=-1))
    df_mesh['Z_img']=Z
    
    return df_mesh
    
def plot_2d(model, columns, quantity, frames, mesh_size=100):
    
    di_f = {}
    for f in frames:
        di_f[f] = get_heatmap_data_6(model, columns, {quantity: f}, mesh_size=mesh_size)
        
    fig = make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': False}, {'is_3d': False}]],
                    subplot_titles=['Color corresponds to Regime', 'Color corresponds to confidence'],
                    )
    
    def add_f_trace(key, col, colorbar_x):
        zmax = 5 if key=="Flow_label" else 1
        showscale = False if key=="Z_img" else True
        return fig.add_trace(go.Heatmap(
                        x=di_f[frames[0]]['Vsg'].values,
                        y=di_f[frames[0]]['Vsl'].values,
                        z=di_f[frames[0]][key].values, 
                        zmin=0, zmax=zmax,
                        colorscale=cs_di[key],
                        customdata=np.stack((di_f[frames[0]]['conf'].values, 
                                             di_f[frames[0]]['Flow Regime'].values), 
                                            axis=-1),
                        hovertemplate=(
                            'Flow Regime: %{customdata[1]} <br>'
                            'Confidence: %{customdata[0]:.3f} <br><extra></extra>'   
                        ), colorbar_x=colorbar_x, showscale=showscale,
                        ), row=1, col=col
                )
    
    def add_m_frames(key, frame):
        zmax = 5 if key=="Flow_label" else 1
        showscale = True if key=="Z_img" else True
        return go.Heatmap(x=di_f[frame]['Vsg'].values,
                          y=di_f[frame]['Vsl'].values,
                          z=di_f[frame][key].values, 
                        zmin=0, zmax=zmax,
                        colorscale=cs_di[key],
                        customdata=np.stack((di_f[frame]['conf'].values, 
                                             di_f[frame]['Flow Regime'].values), 
                                             axis=-1),
                        hovertemplate=(
                            'Flow Regime: %{customdata[1]} <br>'
                            'Confidence: %{customdata[0]:.3f} <br><extra></extra>'   
                            )
                        )        
    
    add_f_trace('Flow_label', 1, colorbar_x=-0.1)
    add_f_trace('conf', 2, colorbar_x=1.0)
    add_f_trace('Z_img', 1, colorbar_x=1.5)
    
    fram = [dict(name=k, 
                 data=[add_m_frames('Flow_label', f), 
                       add_m_frames('conf', f), 
                       add_m_frames('Z_img', f)],
                 traces=[0, 1, 2]) for k, f in enumerate(frames)]
              
    fig.update(frames=fram)
    
    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }
    
    sliders = [{
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(frames[k]),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]
    
    fig.update_layout(
             title=f'{quantity} influence on Flow Pattern',
             width=900,
             height=600,
             scene=dict(
                        zaxis=dict(range=[frames.min(), 1*frames.max()], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),                  
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(2000)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )
    fig.update_xaxes(type="log", title="VsL (ft/s)")
    fig.update_yaxes(type="log")

    fig.show()   

def plot_3d(model, columns, quantity, frames, mesh_size=100, display="Flow Regime"):
    
    di_f = {}
    for f in frames:
        di_f[f] = get_heatmap_data_6(model, columns, {quantity: f}, mesh_size=mesh_size)
        
    if display=="Flow_label":
        cmax=5
        cbar = dict(tickmode="array",
                    tickvals=[0, 1, 2, 3, 4, 5],
                    ticktext=["Annular", "Dispersed", "Intermittent",
                              "Stratified Wavy", "Stratified Smooth", "Bubbly"],
                    ticks="outside"
            )
    else:
        cmax=1
        cbar=dict()  
    
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=frame*np.ones((mesh_size, mesh_size)), 
        surfacecolor=di_f[frame][display].values.reshape(mesh_size, mesh_size).T,
        cmin=0, cmax=cmax,
        colorscale='RdBu',
        customdata=np.stack((di_f[frame]['conf'].values.reshape(mesh_size, mesh_size), 
                             di_f[frame]['Flow Regime'].replace(di).values.reshape(mesh_size, mesh_size)), 
                             axis=-1),
        hovertemplate=(
            'Flow Regime: %{customdata[1]} <br>'
            'Confidence: %{customdata[0]:.3f} <br><extra></extra>'   
        ),            
        colorbar=cbar
    ),
    name=str(frame) # you need to name the frame for the animation to behave properly
    )
    for frame in frames])
    
    fig.add_trace(go.Surface( 
        z=frames[0]*np.ones((mesh_size, mesh_size)),
        surfacecolor=di_f[frames[0]][display].values.reshape(mesh_size, mesh_size).T, 
        cmin=0, cmax=cmax,
        colorscale='RdBu',
        customdata=np.stack((di_f[frames[0]]['conf'].values.reshape(mesh_size, mesh_size), 
                             di_f[frames[0]]['Flow Regime'].replace(di).values.reshape(mesh_size, mesh_size)), 
                            axis=-1),
        hovertemplate=(
            'Flow Regime: %{customdata[1]} <br>'
            'Confidence: %{customdata[0]:.3f} <br><extra></extra>'   
        ),        
        colorbar=cbar
        )
    )

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(frames[k]),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    name = 'eye = (x:0., y:0., z:2.5)'
    camera = dict(
        eye=dict(x=-1., y=-1., z=1.5)
    )

    fig.update_layout(scene_camera=camera, title=name)
    
    # Layout
    fig.update_layout(
             title='Angles influence on Flow Pattern',
             width=600,
             height=600,
             scene=dict(
                        zaxis=dict(range=[frames.min(), 1.1*frames.max()], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )
    fig.show()
    
    
    