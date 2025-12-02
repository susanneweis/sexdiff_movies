
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
import matplotlib.pyplot as plt

def main(): 

    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    nn = 3
    
    results_path = f"{base_path}/results_run_sTOPF"
    results_out_path = f"{base_path}/results_run_sTOPF_{proj}/results_{nn_mi}"

    clust_data_path = f"{results_out_path}/individual_expression_all_nn{nn}_diff_MI_wide.csv" 

    clust_data = pd.read_csv(clust_data_path)
    
    # Separate sex and features
    sex = clust_data['sex']
    X = clust_data.drop(columns=['sex'])
    X = X.drop(columns=['subject'])

    #    Replace inf with NaN just in case
    X = X.replace([np.inf, -np.inf], np.nan)

    # Simple NaN handling: drop any columns that contain NaNs
    # (you can replace this by imputation if needed)
    X = X.dropna(axis=1, how='any')

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # HDBSCAN clustering
    # You can tune min_cluster_size etc.
    # clusterer = hdbscan.HDBSCAN(
    #     min_cluster_size=10,
    #     min_samples=None,
    #     metric='euclidean'
    # )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=1,
        cluster_selection_epsilon=0.1
    )
    cluster_labels = clusterer.fit_predict(X_scaled)   # -1 = noise

    # UMAP to 2D
    # reducer = umap.UMAP(
    #     n_components=2,
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     metric='euclidean',
    #     random_state=42
    # )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.0,   
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    # 5. Put everything into one DataFrame for plotting
    plot_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'cluster': cluster_labels,
        'sex': sex.loc[X.index].values  # align index just in case
    }, index=X.index)


    # 6. Plot 1: Color = sex, marker shape = cluster
    plt.figure(figsize=(8, 6))

    sex_colors = {
        'female': 'tab:red',
        'male': 'tab:blue'
    }

    # Define marker per cluster (noise = -1)
    unique_clusters = sorted(plot_df['cluster'].unique())
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']  # recycled if many clusters
    cluster_marker_map = {
        c: ('x' if c == -1 else markers[i % len(markers)])
        for i, c in enumerate(unique_clusters)
    }

    for (cl, sx), sub in plot_df.groupby(['cluster', 'sex']):
        plt.scatter(
            sub['UMAP1'],
            sub['UMAP2'],
            label=f'Cluster {cl}, {sx}',
            alpha=0.8,
            s=40,
            marker=cluster_marker_map[cl],
            c=sex_colors.get(sx, 'gray')
        )

    clust_o_file = f"{results_out_path}/cluter_individual_expression_{nn}_fig1.png" 

    plt.title('UMAP projection – color = sex, marker = cluster (HDBSCAN)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.savefig(clust_o_file, bbox_inches='tight',dpi=300)
    plt.close()

    # Plot 2: Color = cluster (for checking structure)
    plt.figure(figsize=(8, 6))

    # Build a discrete colormap for clusters
    unique_clusters = sorted(plot_df['cluster'].unique())
    # Assign colors; noise = -1 will be gray
    base_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 3)))
    cluster_color_map = {}
    color_idx = 0
    for c in unique_clusters:
        if c == -1:
            cluster_color_map[c] = 'lightgray'
        else:
            cluster_color_map[c] = base_colors[color_idx]
            color_idx += 1

    for c, sub in plot_df.groupby('cluster'):
        plt.scatter(
            sub['UMAP1'],
            sub['UMAP2'],
            label=f'Cluster {c}' if c != -1 else 'Noise',
            alpha=0.8,
            s=40,
            c=[cluster_color_map[c]]
        )

    clust_o_file = f"{results_out_path}/cluter_individual_expression_{nn}_fig2.png" 

    plt.title('UMAP projection – color = HDBSCAN cluster')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.savefig(clust_o_file, bbox_inches='tight',dpi=300)
    plt.close()



# Execute script
if __name__ == "__main__":
    main()