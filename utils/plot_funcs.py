import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def plot_training_curves(history):
    """3-panel VAE training curves: total loss, reconstruction, KL"""
    epochs = range(1, len(history['train_tloss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # total loss panel — the one we actually watch during training
    axes[0].plot(epochs, history['train_tloss'], label='Train', color='blue')
    axes[0].plot(epochs, history['test_tloss'], label='Test', color='orange', linestyle='--')
    axes[0].set_title('Total Loss (CE + Beta*KL)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # reconstruction panel
    axes[1].plot(epochs, history['train_recon_loss'], label='Train', color='green')
    axes[1].plot(epochs, history['test_recon_loss'], label='Test', color='red', linestyle='--')
    axes[1].set_title('Reconstruction Loss (CE Only)')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL panel — we want this to rise as beta anneals, not collapse to 0
    axes[2].plot(epochs, history['train_kl_loss'], label='Train', color='purple')
    axes[2].plot(epochs, history['test_kl_loss'], label='Test', color='brown', linestyle='--')
    axes[2].set_title('KL Divergence')
    axes[2].set_xlabel('Epochs')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_elbow_method(latents, max_k=10):
    """standalone elbow plot — runs KMeans internally. see also plot_elbow() which takes precomputed metrics"""
    inertias = []
    k_range = range(1, max_k + 1)

    print(f"Running Elbow Method for k=1 to {max_k}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(latents)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()


def plot_tsne_umap(pre_covid_umap, post_covid_umap,
                   pre_covid_tsne, post_covid_tsne,
                   alpha=0.5, s=1):
    """side-by-side UMAP vs t-SNE with pre/post covid colouring"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # we use rasterized=True because scatter plots with 30k+ points are painfully slow without it
    kwargs = {'alpha': alpha, 's': s, 'rasterized': True}
    ax[0].scatter(pre_covid_umap[:, 0], pre_covid_umap[:, 1],
                  label='Pre-COVID', c='grey', **kwargs)
    ax[0].scatter(post_covid_umap[:, 0], post_covid_umap[:, 1],
                  label='Post-COVID', c='red', **kwargs)
    ax[0].legend(markerscale=5)  # markers are tiny at s=1, so we scale up in the legend
    ax[0].set_title(f'UMAP (n={len(pre_covid_umap)+len(post_covid_umap)})')
    ax[0].set_aspect('equal', adjustable='box')

    # same thing for t-SNE — rest of the plots follow this pattern
    ax[1].scatter(pre_covid_tsne[:, 0], pre_covid_tsne[:, 1],
                  label='Pre-COVID', c='grey', **kwargs)
    ax[1].scatter(post_covid_tsne[:, 0], post_covid_tsne[:, 1],
                  label='Post-COVID', c='blue', **kwargs)
    ax[1].legend(markerscale=5)
    ax[1].set_title('t-SNE')
    ax[1].set_aspect('equal', adjustable='box')

    plt.show()


def plot_elbow(k_range, inertias, silhouettes, n_clusters=None, best_k=None):
    """elbow + silhouette side by side — takes precomputed metrics from find_optimal_k()"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range, inertias, 'b-o')
    if n_clusters:
        axes[0].axvline(x=n_clusters, color='r', linestyle='--', label=f'k={n_clusters}')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].set_title('Elbow Method')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, silhouettes, 'g-o')
    if n_clusters:
        axes[1].axvline(x=n_clusters, color='r', linestyle='--', label=f'k={n_clusters}')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title(f'Silhouette (best k={best_k})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_embeddings(embeddings, colors, cmap='viridis', label='Value',
                    title='Embeddings', alpha=0.5, s=1, figsize=(10, 8)):
    """generic single-panel scatter — we colour by whatever you pass in"""
    fig, ax = plt.subplots(figsize=figsize)
    kwargs = {'alpha': alpha, 's': s, 'rasterized': True}

    sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap=cmap, **kwargs)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_title(f'{title} (n={len(embeddings)})')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


# the remaining plot functions all follow the same scatter pattern as above.
# we skip repetitive comments — see plot_tsne_umap() and plot_embeddings() for
# the annotated versions.

def plot_clusters_and_years(embeddings, labels, years, n_clusters, silhouette=None,
                            title='Latent Space', alpha=0.5, s=1, figsize=(14, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    kwargs = {'alpha': alpha, 's': s, 'rasterized': True}

    sc1 = axes[0].scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', **kwargs)
    plt.colorbar(sc1, ax=axes[0], label='Cluster')
    sil_str = f', sil={silhouette:.3f}' if silhouette else ''
    axes[0].set_title(f'Clusters (k={n_clusters}{sil_str})')
    axes[0].set_aspect('equal', adjustable='box')

    sc2 = axes[1].scatter(embeddings[:, 0], embeddings[:, 1], c=years, cmap='viridis', **kwargs)
    plt.colorbar(sc2, ax=axes[1], label='Year')
    axes[1].set_title('Years')
    axes[1].set_aspect('equal', adjustable='box')

    plt.suptitle(f'{title} (n={len(embeddings)})')
    plt.tight_layout()
    plt.show()


def plot_umap_tsne(umap_emb, tsne_emb, colors, cmap='viridis', label='Year',
                   alpha=0.5, s=1, figsize=(14, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    kwargs = {'alpha': alpha, 's': s, 'rasterized': True}

    sc1 = axes[0].scatter(umap_emb[:, 0], umap_emb[:, 1], c=colors, cmap=cmap, **kwargs)
    plt.colorbar(sc1, ax=axes[0], label=label)
    axes[0].set_title(f'UMAP (n={len(umap_emb)})')
    axes[0].set_aspect('equal', adjustable='box')

    sc2 = axes[1].scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=colors, cmap=cmap, **kwargs)
    plt.colorbar(sc2, ax=axes[1], label=label)
    axes[1].set_title('t-SNE')
    axes[1].set_aspect('equal', adjustable='box')

    plt.show()


def plot_pre_post_covid(umap_emb, tsne_emb, years, covid_year=2020, alpha=0.5, s=1, figsize=(14, 6)):
    pre_mask = years < covid_year
    post_mask = years >= covid_year

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    kwargs = {'alpha': alpha, 's': s, 'rasterized': True}

    axes[0].scatter(umap_emb[pre_mask, 0], umap_emb[pre_mask, 1], c='grey', label='Pre-COVID', **kwargs)
    axes[0].scatter(umap_emb[post_mask, 0], umap_emb[post_mask, 1], c='red', label='Post-COVID', **kwargs)
    axes[0].legend(markerscale=5)
    axes[0].set_title(f'UMAP (n={len(umap_emb)})')
    axes[0].set_aspect('equal', adjustable='box')

    axes[1].scatter(tsne_emb[pre_mask, 0], tsne_emb[pre_mask, 1], c='grey', label='Pre-COVID', **kwargs)
    axes[1].scatter(tsne_emb[post_mask, 0], tsne_emb[post_mask, 1], c='blue', label='Post-COVID', **kwargs)
    axes[1].legend(markerscale=5)
    axes[1].set_title('t-SNE')
    axes[1].set_aspect('equal', adjustable='box')

    plt.show()


def plot_full_analysis(embeddings, labels, years, metrics, n_clusters, silhouette,
                       title='Analysis', alpha=0.5, s=1, figsize=(16, 10)):
    """2x2 grid: elbow, silhouette, cluster scatter, year scatter — the kitchen sink plot"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    kwargs = {'alpha': alpha, 's': s, 'rasterized': True}

    # top row: clustering diagnostics
    axes[0, 0].plot(metrics['k_range'], metrics['inertias'], 'b-o')
    axes[0, 0].axvline(x=n_clusters, color='r', linestyle='--', label=f'k={n_clusters}')
    axes[0, 0].set_xlabel('k')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(metrics['k_range'], metrics['silhouettes'], 'g-o')
    axes[0, 1].axvline(x=n_clusters, color='r', linestyle='--', label=f'k={n_clusters}')
    axes[0, 1].set_xlabel('k')
    axes[0, 1].set_ylabel('Silhouette')
    axes[0, 1].set_title(f'Silhouette (best k={metrics["best_k"]})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # bottom row: the actual embeddings
    sc1 = axes[1, 0].scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', **kwargs)
    plt.colorbar(sc1, ax=axes[1, 0], label='Cluster')
    axes[1, 0].set_title(f'Clusters (k={n_clusters}, sil={silhouette:.3f})')
    axes[1, 0].set_aspect('equal', adjustable='box')

    sc2 = axes[1, 1].scatter(embeddings[:, 0], embeddings[:, 1], c=years, cmap='viridis', **kwargs)
    plt.colorbar(sc2, ax=axes[1, 1], label='Year')
    axes[1, 1].set_title('Years')
    axes[1, 1].set_aspect('equal', adjustable='box')

    plt.suptitle(f'{title} (n={len(embeddings)})')
    plt.tight_layout()
    plt.show()