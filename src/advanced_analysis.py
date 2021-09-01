'''Utility functions for advanced analysis'''

import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_explained_variance(pc, explained, save=True):
    
    '''
    Generate a plot representing the explained variance
    as a function of the principal components kept.
    The figure is saved in Plots/DimensionalityReduction
    
    Input: 
        pc : number of principal components
        explained : the explained variance of the PCs
    
    Output:
        Plotly figure of the explained variance
    '''
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pc, 
                             y=explained,
                             mode='lines',
                             name='Single Component'))
    
    fig.add_trace(go.Scatter(x=pc, y=explained.cumsum(),
                        name='Cumulative'))

    fig.update_layout(
        title="PCA results",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance",
    )
    if save:
        fig.write_image("plots/dimensionality_reduction/PCA_variance.png")
    
    return fig.show()


def plot_pca(df_pca, components, save=True):
    
    '''
    Generate the 2D plot between 2 principal components.
    The figure is saved in Plots/DimensionalityReduction
    
    Input: 
    
        df_pca : dataframe of the principal components
        components : tuple [int int] index of the PCs
    Output:
    
        Plotly figure of the selected components
    '''
    
    assert len(components)==2, "Only 2 dimension can be used for this plot"

    fig = px.scatter(df_pca, x=f'PC {components[0]}', y=f'PC {components[1]}',
                     color='Flow Pattern',                   
                     hover_data={f'PC {components[0]}':False,
                                 f'PC {components[1]}':False,
                                 'Flow Pattern':True,
                                 'Index':True})

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_layout(width=800,
                      height=600,
                      legend={'itemsizing': 'constant'})
    
    fig.update_layout(legend=dict(font = dict(family = "Courier",
                                              size = 15, color = "black")),
                      legend_title = dict(font = dict(family = "Flow Pattern", size = 15)))
    if save:
        fig.write_image("plots/dimensionality_reduction"
                        f"/PCA_{components[0]}{components[1]}.png", 
                        scale=2)

    return fig.show()

def plot_tsne(X_scaled, s_target, components, 
              perplexity=150, save=True):
    
    '''
    Generate the t-SNE plot for a given number
    of components (2 or 3). Perplexity can also
    be changed.
    Plot is saved in Plots/DimensionalityReduction
    
    Input:
        X_scaled : the standardized features
        s_target : the target label
        components : int number of dimensions to keep
        perplexity : the iterations performed by t-SNE
    
    Output:
        t-SNE plot
    '''
    
    method = TSNE(n_components=components, 
                  init='pca', perplexity=perplexity, 
                  random_state=42)
    
    #Fit transform the data keeping only the desired components
    X_tSNE = method.fit_transform(X_scaled)
    df_tSNE = pd.DataFrame({f't-SNE {c}' : X_tSNE[:, c] 
                            for c in range(components)})
    df_tSNE['Flow Pattern'] = s_target
    df_tSNE["Index"] = df_tSNE.index
    
    #3D plot
    if components==3:
        size=3
        width=1
        
        fig = px.scatter_3d(df_tSNE, x='t-SNE 0', y='t-SNE 1', z='t-SNE 2',
                  color='Flow Pattern', hover_data={'t-SNE 0':False,
                                                    't-SNE 1':False,
                                                    't-SNE 2':False,
                                                    'Flow Pattern':True,
                                                    'Index':True})
    #2D plot
    else:
        size=6
        width=1

        fig = px.scatter(df_tSNE, x='t-SNE 0', y='t-SNE 1', 
                   color='Flow Pattern', hover_data={'t-SNE 0':False,
                                                     't-SNE 1':False,
                                                     'Flow Pattern':True,
                                                     'Index':True})
    
    #Stylize the plot
    fig.update_traces(marker=dict(size=size, line=dict(width=width,
                                  color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    
    
    fig.update_layout(legend=dict(font=dict(family = "Courier", 
                                            size = 15, 
                                            color = "black")),
                      legend_title = dict(font = dict(family = "Flow Pattern", size = 15)))  
    
    fig.update_layout(width=800,
                      height=600,legend={'itemsizing': 'constant'})
    if save:
        fig.write_image("plots/dimensionality_reduction"
                        f"/tSNE_{components}.png", scale=2)
    
    return fig.show()

def plot_paracords(df, save=True):
    
    '''
    Plot the paracrods graph for a dataset
    
    Input:
        df : the dataset to analyze
        
    Output:
        Parallel coordinates plot
    '''
    
    di = {0: 'Annular',           1: 'Dispersed Bubbly', 
          2: 'Intermittent',      3: 'Stratified Wavy', 
          4: 'Stratified Smooth', 5: 'Bubbly'}
    
    #Select features of interest
    dimensions = list([dict(label='log(FrL)', values=np.log10(df['FrL'])),
                       dict(label='log(FrG)', values=np.log10(df['FrG'])),
                       dict(label='Eo', values=np.log10(df['Eo'])),
                       dict(label='X_LM', values=np.log10(df['X_LM'])),
                       dict(label='Ang', values=df['Ang']),
                       dict(range=[0,df['Flow_label'].max()],
                            tickvals = list(di.keys()), 
                            ticktext = list(di.values()),
                            label='Flow Regime', values=df['Flow_label'])])

    fig = go.Figure(data=go.Parcoords(line = dict(color = df['Flow_label'], 
                                colorscale = 'RdBu'), dimensions=dimensions))
    
    fig.update_layout(width=800,
                  height=600,
                  legend={'itemsizing': 'constant'})

    
    if save:
        fig.write_image("plots/others/paracords.png", scale=2)      
    
    return fig.show()