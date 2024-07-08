import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.cm import viridis
import plotly.graph_objects as go

def prepare_data(data):
    close_prices = data.xs('Close', level='Feature', axis=1)
    returns = close_prices.pct_change().dropna()
    return returns

def perform_pca(returns):
    mean = np.mean(returns, axis=0)
    std = np.std(returns, axis=0)
    standardized_returns = (returns - mean) / std

    cov_matrix = np.cov(standardized_returns.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var

    pca_results = np.dot(standardized_returns, eigenvectors)

    return pca_results, explained_variance_ratio, eigenvectors

def plot_explained_variance(explained_variance_ratio):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
             np.cumsum(explained_variance_ratio), 'ro-')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

def plot_pca_results_3d(pca_results, returns, eigenvectors, explained_variance_ratio):
    max_range = np.max([pca_results[:, 0].max() - pca_results[:, 0].min(),
                        pca_results[:, 1].max() - pca_results[:, 1].min(),
                        pca_results[:, 2].max() - pca_results[:, 2].min()])
    
    date_nums = np.array([d.toordinal() for d in returns.index])
    color_vals = (date_nums - date_nums.min()) / (date_nums.max() - date_nums.min())
    colors = [to_hex(viridis(val)) for val in color_vals]

    scatter = go.Scatter3d(
        x=pca_results[:, 0],
        y=pca_results[:, 1],
        z=pca_results[:, 2],
        mode='markers',
        marker=dict(size=3, color=colors, opacity=0.8),
        text=[f"Date: {date}<br>PC1: {pc1:.2f}<br>PC2: {pc2:.2f}<br>PC3: {pc3:.2f}" 
              for date, pc1, pc2, pc3 in zip(returns.index, pca_results[:, 0], pca_results[:, 1], pca_results[:, 2])],
        hoverinfo='text',
        name='Data Points'
    )

    vector_colors = ['red', 'green', 'blue']
    pca_vectors = []
    for i in range(3):
        vector = eigenvectors[:, i] * explained_variance_ratio[i] * max_range
        pca_vectors.append(go.Scatter3d(
            x=[0, vector[0]],
            y=[0, vector[1]],
            z=[0, vector[2]],
            mode='lines+text',
            line=dict(color=vector_colors[i], width=6),
            text=['', f'PC{i+1}'],
            textposition='top center',
            name=f'PC{i+1} ({explained_variance_ratio[i]:.2%})'
        ))

    traces = [scatter] + pca_vectors

    layout = go.Layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title='3D PCA of Stock Returns',
        width=900,
        height=700,
        legend=dict(x=0.7, y=0.9)
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.write_html("pca_3d_plot.html")
    print("3D PCA plot saved as 'pca_3d_plot.html'. Please open this file in a web browser.")

def plot_pca_results_2d(pca_results, returns, eigenvectors, explained_variance_ratio, filename=None):
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5, label='Data Points')

    for i in range(2):
        plt.arrow(0, 0, eigenvectors[0, i]*explained_variance_ratio[i]*20, 
                  eigenvectors[1, i]*explained_variance_ratio[i]*20, 
                  head_width=0.5, head_length=0.5, fc='r', ec='r',
                  label=f'PC{i+1}' if i==0 else "_nolegend_")
    
    plt.title('2D PCA of Stock Returns with Principal Components')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    for i, txt in enumerate(returns.index):
        if i % 50 == 0:
            plt.annotate(txt, (pca_results[i, 0], pca_results[i, 1]))
    
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    if filename:
        plt.savefig(filename)
    plt.show()
