import torch
import numpy as np
import umap  # non-linear dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .plot_funcs import (
    plot_elbow,
    plot_embeddings,
    plot_clusters_and_years,
    plot_umap_tsne,
    plot_pre_post_covid,
    plot_full_analysis)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_latents(model, loader, device=device):
  """we always pull the mean latent because that's the representation we score later"""
  model.eval()
  use_amp = device.type == 'cuda'
  latents, years, ids, months, seasons = [], [], [], [], []
  # we use inference_mode here because there's no reason to keep autograd alive
  # during latent extraction, and it cuts some overhead on long sequence batches
  with torch.inference_mode():
    for x, meta in loader:
        x = x.to(device).float()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _, mu, _ = model(x)
        # we keep the tensors on device until the end so we don't pay a host sync
        # every batch when extracting a full subtype
        latents.append(mu)
        years.extend(meta[0].numpy())
        ids.extend(meta[1])
        if len(meta) > 2:
            months.extend(meta[2])
            seasons.extend(meta[3])
  all_latents = torch.cat(latents).float().cpu().numpy()  # one sync instead of N
  result = (all_latents, np.array(years), np.array(ids))
  if months:
      result = result + (months, seasons)
  return result


def balanced_pre_post_split(latents: np.ndarray, years: np.ndarray, covid_year: int = 2020,
                            random_state: int = 42, exclude_covid: bool = True,
                            seasons=None):
  """splits latents into pre/post COVID with equal sequences per season.
  we downsample every season to the smallest season's count so no single
  season dominates the embedding. exclude_covid drops 2020-21 by default."""
  rng = np.random.default_rng(random_state)
  seasons = np.array(seasons) if seasons is not None else None

  if exclude_covid:
      keep = ~((years >= covid_year) & (years <= covid_year + 1))
      latents, years = latents[keep], years[keep]
      if seasons is not None:
          seasons = seasons[keep]

  # per-season balancing: subsample every season to the smallest season's count
  if seasons is not None:
      unique_seasons = sorted(set(seasons))
      min_per_season = min(np.sum(seasons == s) for s in unique_seasons)
      keep_idx = []
      for s in unique_seasons:
          s_idx = np.where(seasons == s)[0]
          chosen = rng.choice(s_idx, size=min_per_season, replace=False)
          keep_idx.extend(chosen)
      keep_idx = sorted(keep_idx)
      latents, years = latents[keep_idx], years[keep_idx]
      n_seasons = len(unique_seasons)
      print(f"Balanced to {min_per_season} sequences × {n_seasons} seasons "
            f"= {len(latents)} total")
  else:
      # fallback: balance pre vs post by total count
      pre_mask = years < covid_year
      n_pre, n_post = pre_mask.sum(), (~pre_mask).sum()
      n_min = min(n_pre, n_post)
      pre_idx = rng.choice(np.where(pre_mask)[0], size=n_min, replace=False)
      post_idx = rng.choice(np.where(~pre_mask)[0], size=n_min, replace=False)
      keep_idx = sorted(np.concatenate([pre_idx, post_idx]))
      latents, years = latents[keep_idx], years[keep_idx]
      print(f"Balanced split: {n_min} pre-COVID, {n_min} post-COVID "
            f"(downsampled from {n_pre}/{n_post})")

  pre_mask = years < covid_year
  return latents[pre_mask], years[pre_mask], latents[~pre_mask], years[~pre_mask]


def umap_embeddings(pre_covid_latents, post_covid_latents, n_neighbors=15, min_dist=0.1, random_state=42):
  """pre/post covid split — we fit on pre-covid only so post-covid structure isn't leaked into the projection"""
  reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
  pre_covid_embeddings = reducer.fit_transform(pre_covid_latents)
  post_covid_embeddings = reducer.transform(post_covid_latents)
  return pre_covid_embeddings, post_covid_embeddings

def tsne_embeddings(pre_covid_latents, post_covid_latents, perplexity=30, emb_dim=2, init='pca'):
    """t-SNE has no .transform() so we concat and split — not ideal but it's how t-SNE works"""
    len_pre = len(pre_covid_latents)
    combined_latents = np.concatenate([pre_covid_latents, post_covid_latents], axis=0)

    reducer = TSNE(n_components=emb_dim, perplexity=perplexity, n_jobs=-1, random_state=42, init=init, learning_rate='auto')
    combined_embeddings = reducer.fit_transform(combined_latents)

    pre_covid_embeddings = combined_embeddings[:len_pre]
    post_covid_embeddings = combined_embeddings[len_pre:]

    return pre_covid_embeddings, post_covid_embeddings

# standalone helpers — simpler versions of the class methods

def umap_embed(latents, n_neighbors=15, min_dist=0.1, random_state=42):
   reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
   return reducer.fit_transform(latents)

def tsne_embed(latents, perplexity=30, emb_dim=2, init='pca'):
   reducer = TSNE(n_components=emb_dim, perplexity=perplexity, n_jobs=-1, random_state=42, init=init, learning_rate='auto')
   return reducer.fit_transform(latents)

def k_means_cluster(embeddings, n_clusters=5, random_state=42, return_model=False):
   model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
   labels = model.fit_predict(embeddings)
   if return_model:
      return labels, model
   return labels
   
class latent_analysis:
  """we bundled the analysis helpers here so the training notebooks stay readable"""

  def __init__(self, model, loader, device=None):
    self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = model
    self.loader = loader
    self.latents, self.years, self.ids = self._extract_latents()
    # we cache embeddings keyed by method+params so we don't recompute
    self.embeddings = {}
    self.labels = None
    self.metrics = {}

  def _extract_latents(self):
    self.model.eval()
    use_amp = self.device.type == 'cuda'
    latents, years, ids = [], [], []
    self.months, self.seasons = [], []
    with torch.inference_mode():
      for x, meta in self.loader:
          x = x.to(self.device).float()
          with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
              _, mu, _ = self.model(x)
          latents.append(mu)
          years.extend(meta[0].numpy())
          ids.extend(meta[1])
          if len(meta) > 2:
              self.months.extend(meta[2])
              self.seasons.extend(meta[3])
    return torch.cat(latents).float().cpu().numpy(), np.array(years), np.array(ids)
    
  def umap_embed(self, n_neighbors=15, min_dist=0.1, random_state=42, force=False):
    key = f'umap_{n_neighbors}_{min_dist}'
    if key not in self.embeddings or force:
        print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        self.embeddings[key] = reducer.fit_transform(self.latents)
    else:
        print("Using cached UMAP embeddings.")
    return self.embeddings[key]

  def tsne_embed(self, perplexity=30, random_state=42, force=False, init='pca'):
    key = f'tsne_{perplexity}'
    if key not in self.embeddings or force:
        print(f"Computing t-SNE (perplexity={perplexity})...")
        reducer = TSNE(n_components=2, perplexity=perplexity, n_jobs=-1,
                      random_state=random_state, init='pca', learning_rate='auto')
        self.embeddings[key] = reducer.fit_transform(self.latents)
    else:
        print("Using cached t-SNE embeddings.")
    return self.embeddings[key]

  def compute_embedding(self, method='umap', **kwargs):
     """convenience dispatcher — just routes to umap_embed or tsne_embed"""
     if method == 'umap':
        return self.umap_embed(**kwargs)
     elif method == 'tsne':
        return self.tsne_embed(**kwargs)
     else:
        raise ValueError(f"Unknown method: {method}")
     
  def find_optimal_k(self, embeddings, max_k=10):
      """brute-force k search — we pick the k with highest silhouette score"""
      k_range = range(2, max_k + 1)
      inertias, silhouettes = [], []

      print(f"Optimal k (2 to {max_k})...")
      for k in k_range:
          kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
          labels = kmeans.fit_predict(embeddings)
          inertias.append(kmeans.inertia_)
          silhouettes.append(silhouette_score(embeddings, labels))

      best_k = list(k_range)[np.argmax(silhouettes)]

      self.metrics = {
          'k_range': list(k_range),
          'inertias': inertias,
          'silhouettes': silhouettes,
          'best_k': best_k
      }
      return self.metrics

  def cluster(self, embeddings, n_clusters='auto', max_k=10):
      if n_clusters == 'auto':
          if not self.metrics:
              self.find_optimal_k(embeddings, max_k)
          n_clusters = self.metrics['best_k']
          print(f"Automatically selected k={n_clusters}")

      kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
      self.labels = kmeans.fit_predict(embeddings)
      self.n_clusters = n_clusters
      self.silhouette = silhouette_score(embeddings, self.labels)

      # season-based silhouette — measures what we actually care about for drift
      self.season_silhouette = self._season_silhouette()

      return self.labels

  def _season_silhouette(self):
      """silhouette on actual season labels rather than KMeans — measures whether
      the latent space naturally separates flu seasons, aligned with drift task"""
      if not self.seasons or len(set(self.seasons)) < 2:
          return None
      label_map = {s: i for i, s in enumerate(sorted(set(self.seasons)))}
      int_labels = np.array([label_map[s] for s in self.seasons])
      return silhouette_score(self.latents, int_labels)
  
  # plotting methods — thin wrappers around plot_funcs.py
  # we keep them here so notebooks can just do `analysis.plot_elbow()` etc.

  def plot_elbow(self, embeddings=None, max_k=10):
    if not self.metrics:
        if embeddings is None:
            raise ValueError("Provide embeddings or run find_optimal_k() first")
        self.find_optimal_k(embeddings, max_k)
    n_clusters = getattr(self, 'n_clusters', None) or self.metrics['best_k']
    plot_elbow(self.metrics['k_range'], self.metrics['inertias'],
               self.metrics['silhouettes'], n_clusters, self.metrics['best_k'])

  def plot_by_year(self, embeddings, title='Embeddings'):
      plot_embeddings(embeddings, self.years, cmap='viridis', label='Year', title=title)

  def plot_by_cluster(self, embeddings, title='Embeddings'):
      plot_embeddings(embeddings, self.labels, cmap='tab10', label='Cluster', title=title)

  def plot_clusters_and_years(self, embeddings, title='Latent Space'):
      plot_clusters_and_years(embeddings, self.labels, self.years,
                              self.n_clusters, self.silhouette, title)

  def plot_umap_tsne(self, color_by='year'):
      umap_emb = self.umap_embed()
      tsne_emb = self.tsne_embed()
      colors = self.years if color_by == 'year' else self.labels
      cmap = 'viridis' if color_by == 'year' else 'tab10'
      label = 'Year' if color_by == 'year' else 'Cluster'
      plot_umap_tsne(umap_emb, tsne_emb, colors, cmap, label)

  def plot_pre_post_covid(self, covid_year=2020, balance=True):
      if balance:
          # we balance so the larger post-COVID group doesn't dominate the projection
          pre_lat, pre_y, post_lat, post_y = balanced_pre_post_split(
              self.latents, self.years, covid_year,
              seasons=self.seasons if self.seasons else None)
          combined = np.concatenate([pre_lat, post_lat])
          combined_years = np.concatenate([pre_y, post_y])
          # we recompute embeddings on balanced data so projections aren't skewed
          umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
          umap_emb = umap_reducer.fit_transform(combined)
          tsne_reducer = TSNE(n_components=2, perplexity=30, n_jobs=-1, random_state=42,
                              init='pca', learning_rate='auto')
          tsne_emb = tsne_reducer.fit_transform(combined)
          plot_pre_post_covid(umap_emb, tsne_emb, combined_years, covid_year)
      else:
          umap_emb = self.umap_embed()
          tsne_emb = self.tsne_embed()
          plot_pre_post_covid(umap_emb, tsne_emb, self.years, covid_year)

  def analyse(self, method='umap', n_clusters='auto', max_k=10, title='Analysis'):
      """the full pipeline: embed -> find k -> cluster -> plot everything"""
      embeddings = self.umap_embed() if method == 'umap' else self.tsne_embed()
      self.find_optimal_k(embeddings, max_k)
      self.cluster(embeddings, n_clusters, max_k)
      if self.season_silhouette is not None:
          print(f"Silhouette — KMeans: {self.silhouette:.4f}, Season: {self.season_silhouette:.4f}")
      plot_full_analysis(embeddings, self.labels, self.years, self.metrics,
                          self.n_clusters, self.silhouette, f'{method.upper()} {title}')
      return embeddings
