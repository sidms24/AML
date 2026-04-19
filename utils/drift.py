import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as _cosine_dist


def _compute_distance(a, b, metric='euclidean'):
    """distance between two centroid vectors"""
    if metric == 'cosine':
        return _cosine_dist(a, b)
    return np.linalg.norm(a - b)


def _season_start_year(season_str):
    """extract the first year from a '2017-18' style season string"""
    return int(str(season_str).split('-')[0])


def _season_month_order(month_int):
    """flu seasons run Oct -> Sep, so we need a custom month ordering"""
    return month_int - 9 if month_int >= 10 else month_int + 3


def compute_drift_scores(latents, years, months=None, seasons=None,
                         max_per_season=None, dist_metric='euclidean', lag=1):
    """computes latent sequence drift scores from latent vectors.
    we call these "latent sequence drift scores" not "antigenic drift scores"
    because they're derived from sequence embeddings, not validated against HI assays.
    max_per_season: cap sequences per season to remove sample-size confound.
    dist_metric: 'euclidean' (L2) or 'cosine' for centroid distance.
    lag: number of seasons back to compare (1 = consecutive, 2 = skip one).
    Perofsky et al. 2024 found lag-2 is stronger for HA epitope distance"""

    if seasons is not None:
        return _drift_by_season(latents, seasons, max_per_season=max_per_season,
                                dist_metric=dist_metric, lag=lag)
    elif months is not None:
        seasons_arr = np.array([
            assign_season(int(y), int(m) if m > 0 else None)
            for y, m in zip(years, months)
        ])
        return _drift_by_season(latents, seasons_arr, max_per_season=max_per_season,
                                dist_metric=dist_metric, lag=lag)
    else:
        return _drift_by_year(latents, years, dist_metric=dist_metric)


def _drift_by_season(latents, seasons, max_per_season=None, seed=42,
                     dist_metric='euclidean', lag=1):
    """groups latents by season, computes centroids, then measures centroid distances.
    lag=1 compares consecutive seasons (default), lag=2 skips one season —
    Perofsky et al. 2024 found 2-season lag is more predictive for HA epitope drift.
    max_per_season: subsample to remove sample-size confound"""
    unique_seasons = sorted(set(seasons), key=_season_sort_key)
    rng = np.random.default_rng(seed)
    centroids = {}
    for s in unique_seasons:
        mask = np.array([ss == s for ss in seasons])
        season_latents = latents[mask]
        if max_per_season and len(season_latents) > max_per_season:
            idx = rng.choice(len(season_latents), max_per_season, replace=False)
            season_latents = season_latents[idx]
        centroids[s] = season_latents.mean(axis=0)

    drift_records = []
    for i in range(lag, len(unique_seasons)):
        s_curr = unique_seasons[i]
        s_prev = unique_seasons[i - lag]
        dist = _compute_distance(centroids[s_curr], centroids[s_prev], dist_metric)
        n_seqs = int(sum(1 for ss in seasons if ss == s_curr))
        drift_records.append({
            'season': s_curr,
            'drift_raw': dist,
            'n_sequences': n_seqs,
            'centroid': centroids[s_curr],
        })

    df = pd.DataFrame(drift_records)
    df['drift_norm'] = _expanding_window_norm(df['drift_raw'])
    return df


def _drift_by_year(latents, years, dist_metric='euclidean'):
    """year-based grouping — kept for backwards compat but season-based is preferred"""
    unique_years = sorted(np.unique(years))
    centroids = {}
    for y in unique_years:
        mask = years == y
        centroids[y] = latents[mask].mean(axis=0)

    drift_records = []
    for i in range(1, len(unique_years)):
        y_curr = unique_years[i]
        y_prev = unique_years[i - 1]
        dist = _compute_distance(centroids[y_curr], centroids[y_prev], dist_metric)
        n_seqs = int((years == y_curr).sum())
        drift_records.append({
            'year': y_curr,
            'season': assign_season(int(y_curr)),
            'drift_raw': dist,
            'n_sequences': n_seqs,
            'centroid': centroids[y_curr],
        })

    df = pd.DataFrame(drift_records)
    df['drift_norm'] = _expanding_window_norm(df['drift_raw'])
    return df


def compute_monthly_drift_scores(latents, years, months, max_per_month=None,
                                 dist_metric='euclidean', lag=1, seed=42):
    """compute drift at month resolution so weekly models can see within-season change.

    we compare each (season, month) centroid against the same calendar month
    from the season `lag` years earlier. that keeps the comparison cleaner than
    matching months that sit at different points in the epidemic curve."""
    months = np.asarray(months, dtype=np.int32)
    years = np.asarray(years)
    valid = months > 0
    if not valid.any():
        raise ValueError('compute_monthly_drift_scores needs valid month labels')

    latents = latents[valid]
    years = years[valid]
    months = months[valid]
    seasons = np.array([assign_season(int(y), int(m)) for y, m in zip(years, months)])

    rng = np.random.default_rng(seed)
    centroids = {}
    counts = {}
    for season in sorted(set(seasons), key=_season_sort_key):
        season_months = sorted(set(months[seasons == season]), key=_season_month_order)
        for month_int in season_months:
            mask = (seasons == season) & (months == month_int)
            month_latents = latents[mask]
            if max_per_month and len(month_latents) > max_per_month:
                idx = rng.choice(len(month_latents), max_per_month, replace=False)
                month_latents = month_latents[idx]
            centroids[(season, int(month_int))] = month_latents.mean(axis=0)
            counts[(season, int(month_int))] = int(mask.sum())

    ordered_keys = sorted(
        centroids,
        key=lambda sm: (_season_start_year(sm[0]), _season_month_order(sm[1])),
    )

    records = []
    for season, month_int in ordered_keys:
        prev_start = _season_start_year(season) - lag
        prev_season = f'{prev_start}-{str(prev_start + 1)[-2:]}'
        prev_key = (prev_season, month_int)
        if prev_key not in centroids:
            continue
        dist = _compute_distance(centroids[(season, month_int)], centroids[prev_key], dist_metric)
        records.append({
            'season': season,
            'month_int': int(month_int),
            'drift_raw': dist,
            'n_sequences': counts[(season, month_int)],
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    order = df['month_int'].map(_season_month_order)
    df = df.assign(_order=order).sort_values(['season', '_order']).reset_index(drop=True)
    df['drift_norm'] = _expanding_window_norm(df['drift_raw'])
    return df.drop(columns='_order')


def compute_early_season_drift_scores(latents, years, months, n_early_months=2,
                                      max_per_season=None, dist_metric='euclidean',
                                      lag=1, seed=42):
    """compute one drift score per season using only the first few season-months.

    this is the cleaner covariate when drift is meant to drive S0, because S0 is
    a season-start quantity. we pool the first `n_early_months` observed months
    within each flu season, then compare that early-season centroid to the same
    early window from `lag` seasons earlier."""
    months = np.asarray(months, dtype=np.int32)
    years = np.asarray(years)
    valid = months > 0
    if not valid.any():
        raise ValueError('compute_early_season_drift_scores needs valid month labels')

    latents = latents[valid]
    years = years[valid]
    months = months[valid]
    seasons = np.array([assign_season(int(y), int(m)) for y, m in zip(years, months)])

    rng = np.random.default_rng(seed)
    centroids = {}
    counts = {}
    unique_seasons = sorted(set(seasons), key=_season_sort_key)

    for season in unique_seasons:
        season_mask = seasons == season
        season_months = sorted(set(months[season_mask]), key=_season_month_order)
        early_months = set(season_months[:n_early_months])
        early_mask = season_mask & np.isin(months, list(early_months))
        season_latents = latents[early_mask]
        if len(season_latents) == 0:
            continue
        if max_per_season and len(season_latents) > max_per_season:
            idx = rng.choice(len(season_latents), max_per_season, replace=False)
            season_latents = season_latents[idx]
        centroids[season] = season_latents.mean(axis=0)
        counts[season] = int(early_mask.sum())

    records = []
    for i in range(lag, len(unique_seasons)):
        s_curr = unique_seasons[i]
        s_prev = unique_seasons[i - lag]
        if s_curr not in centroids or s_prev not in centroids:
            continue
        dist = _compute_distance(centroids[s_curr], centroids[s_prev], dist_metric)
        records.append({
            'season': s_curr,
            'drift_raw': dist,
            'drift_window_months': n_early_months,
            'n_sequences': counts[s_curr],
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df['drift_norm'] = _expanding_window_norm(df['drift_raw'])
    return df


def _expanding_window_norm(drift_raw):
    """expanding-window min-max normalisation — critical for avoiding data leakage.
    we only ever look backwards in time so future seasons can't influence the scale.

    caveat: early seasons have a narrow normalisation window (e.g. the 3rd season
    is normalised over only 3 raw values) which makes their normalised scores more
    volatile than later seasons. this is inherent to the expanding-window approach
    and should be noted when interpreting early-season drift scores."""
    drift_norm = []
    for i in range(len(drift_raw)):
        window = drift_raw.iloc[:i + 1]
        mn, mx = window.min(), window.max()
        if mx > mn:
            drift_norm.append((drift_raw.iloc[i] - mn) / (mx - mn))
        else:
            # first data point or all identical — 0.5 is a neutral default
            drift_norm.append(0.5)
    return drift_norm


def _season_sort_key(season_str):
    """sorts '2017-18' style season strings by start year"""
    try:
        return int(season_str.split('-')[0])
    except (ValueError, IndexError):
        return 0


def assign_season(year, month=None):
    """maps a collection date to a flu season. flu seasons run Oct-Sep so
    Oct/Nov/Dec of year Y belongs to season Y-(Y+1), everything else to (Y-1)-Y"""
    if month is not None and month >= 10:
        return f'{year}-{str(year + 1)[-2:]}'
    else:
        return f'{year - 1}-{str(year)[-2:]}'


def compute_latent_diversity(latents, seasons):
    """per-season latent diversity — mean L2 distance from centroid.
    inspired by Perofsky et al. 2024 who use Shannon entropy of LBI bins
    to capture whether one dominant clade or many co-circulating lineages exist.
    our analog: how spread out are sequences in latent space within each season?
    high diversity = multiple distinct variants co-circulating"""
    unique_seasons = sorted(set(seasons), key=_season_sort_key)
    records = []
    for s in unique_seasons:
        mask = np.array([ss == s for ss in seasons])
        vecs = latents[mask]
        if len(vecs) < 2:
            continue
        centroid = vecs.mean(axis=0)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        records.append({
            'season': s,
            'latent_spread': dists.mean(),
            'latent_spread_std': dists.std(),
            'latent_var_trace': np.trace(np.cov(vecs.T)),
            'n_sequences': len(vecs),
        })
    return pd.DataFrame(records)


def validate_latent_drift(latent_drift_df, hamming_drift_df, on='season',
                          latent_col='drift_raw', hamming_col=None):
    """compare latent drift scores against a sequence-level baseline (e.g. epitope
    Hamming distance) to check whether the autoencoder captures biologically
    meaningful variation. reports Spearman rho + bootstrap 95% CI.

    this is NOT a validation against the antigenic gold standard (HI titers) —
    we don't have ferret sera data. it only tests whether latent drift tracks
    raw sequence change. a low correlation would suggest the autoencoder is
    capturing noise or batch effects rather than genuine evolutionary signal.

    hamming_col: column name in hamming_drift_df (auto-detects 'drift_raw' or 'hamming_raw')"""
    from scipy.stats import spearmanr

    # auto-detect the Hamming column name if not specified
    if hamming_col is None:
        if 'drift_raw' in hamming_drift_df.columns:
            hamming_col = 'drift_raw'
        elif 'hamming_raw' in hamming_drift_df.columns:
            hamming_col = 'hamming_raw'
        else:
            raise KeyError(f'Cannot find drift column in hamming_drift_df. '
                           f'Columns: {list(hamming_drift_df.columns)}')

    merged = latent_drift_df[[on, latent_col]].merge(
        hamming_drift_df[[on, hamming_col]].rename(columns={hamming_col: 'hamming_val'}),
        on=on,
    ).dropna()

    if len(merged) < 4:
        print(f'WARNING: only {len(merged)} overlapping seasons — too few for validation')
        return None

    rho, p = spearmanr(merged[latent_col], merged['hamming_val'])

    # bootstrap 95% CI
    rng = np.random.default_rng(42)
    boot_rhos = []
    for _ in range(1000):
        idx = rng.choice(len(merged), size=len(merged), replace=True)
        if len(set(idx)) < 3:
            continue
        r, _ = spearmanr(merged[latent_col].iloc[idx],
                         merged['hamming_val'].iloc[idx])
        if not np.isnan(r):
            boot_rhos.append(r)
    ci = np.percentile(boot_rhos, [2.5, 97.5]) if boot_rhos else (np.nan, np.nan)

    print(f'Latent vs Hamming drift validation:')
    print(f'  Spearman rho = {rho:.3f} (p = {p:.4f})')
    print(f'  95% bootstrap CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
    print(f'  n = {len(merged)} seasons')

    return {
        'rho': rho, 'p': p,
        'ci_lo': ci[0], 'ci_hi': ci[1],
        'n_seasons': len(merged),
        'merged': merged,
    }


def compute_blended_drift(season_summary, drift_h1n1, drift_h3n2, exclude_covid=True):
    """merges per-subtype drift scores with CDC subtype proportions to get a single
    blended drift score per season. we weight by actual subtype dominance so a big
    H3N2 drift in an H1N1-dominant season doesn't artificially inflate the signal"""

    df = season_summary[['season', 'pct_h1n1', 'pct_h3n2',
                          'hosp_rate_overall', 'covid_excluded']].copy()

    # build lookup dicts so we can map season -> drift score
    h1_lookup = drift_h1n1.set_index('season')['drift_norm'].to_dict()
    h3_lookup = drift_h3n2.set_index('season')['drift_norm'].to_dict()
    h1_raw = drift_h1n1.set_index('season')['drift_raw'].to_dict()
    h3_raw = drift_h3n2.set_index('season')['drift_raw'].to_dict()

    df['drift_h1n1'] = df['season'].map(h1_lookup)
    df['drift_h3n2'] = df['season'].map(h3_lookup)
    df['drift_h1n1_raw'] = df['season'].map(h1_raw)
    df['drift_h3n2_raw'] = df['season'].map(h3_raw)

    # if subtype proportions are missing we fall back to equal weighting instead of
    # zeroing them out — zero would silently kill the drift signal for that season
    has_props = df['pct_h1n1'].notna() & df['pct_h3n2'].notna()
    prop_h1 = df['pct_h1n1'].fillna(0.5)
    prop_h3 = df['pct_h3n2'].fillna(0.5)

    # CDC proportions should already sum to ~1 across H1N1/H3N2 after B is excluded,
    # but we renormalise just in case to keep the blend interpretable.
    prop_sum = prop_h1 + prop_h3
    prop_h1 = np.where(prop_sum > 0, prop_h1 / prop_sum, 0.5)
    prop_h3 = np.where(prop_sum > 0, prop_h3 / prop_sum, 0.5)

    # do not treat missing subtype drift as true zero novelty. instead,
    # renormalise over the subtype signals that actually exist for that season.
    has_h1 = df['drift_h1n1'].notna()
    has_h3 = df['drift_h3n2'].notna()
    eff_w_h1 = np.where(has_h1, prop_h1, 0.0)
    eff_w_h3 = np.where(has_h3, prop_h3, 0.0)
    eff_w_sum = eff_w_h1 + eff_w_h3
    eff_w_h1 = np.where(eff_w_sum > 0, eff_w_h1 / eff_w_sum, np.nan)
    eff_w_h3 = np.where(eff_w_sum > 0, eff_w_h3 / eff_w_sum, np.nan)

    df['drift_blended'] = (
        eff_w_h1 * df['drift_h1n1'] +
        eff_w_h3 * df['drift_h3n2']
    )
    df['blend_method'] = np.select(
        [
            has_props & has_h1 & has_h3,
            has_h1 & ~has_h3,
            ~has_h1 & has_h3,
        ],
        [
            'observed',
            'h1_only',
            'h3_only',
        ],
        default='equal_weight_fallback',
    )

    df = df.dropna(subset=['hosp_rate_overall', 'drift_blended'])
    if exclude_covid:
        df = df[df['covid_excluded'] == False].copy()

    return df


def compute_blended_monthly_drift(subtype_monthly, drift_h1n1_monthly,
                                  drift_h3n2_monthly):
    """blend monthly subtype-specific drift into one monthly drift signal.

    we still weight H1/H3 by observed subtype mix, just at month resolution
    instead of flattening the whole season into one number."""
    df = subtype_monthly[['season', 'month_int', 'pct_h1n1', 'pct_h3n2']].copy()

    h1_lookup = drift_h1n1_monthly.set_index(['season', 'month_int'])['drift_norm'].to_dict()
    h3_lookup = drift_h3n2_monthly.set_index(['season', 'month_int'])['drift_norm'].to_dict()

    df['drift_h1n1'] = [h1_lookup.get((s, m)) for s, m in zip(df['season'], df['month_int'])]
    df['drift_h3n2'] = [h3_lookup.get((s, m)) for s, m in zip(df['season'], df['month_int'])]

    prop_h1 = df['pct_h1n1'].fillna(0.5)
    prop_h3 = df['pct_h3n2'].fillna(0.5)
    prop_sum = prop_h1 + prop_h3
    prop_h1 = np.where(prop_sum > 0, prop_h1 / prop_sum, 0.5)
    prop_h3 = np.where(prop_sum > 0, prop_h3 / prop_sum, 0.5)

    has_h1 = df['drift_h1n1'].notna()
    has_h3 = df['drift_h3n2'].notna()
    eff_w_h1 = np.where(has_h1, prop_h1, 0.0)
    eff_w_h3 = np.where(has_h3, prop_h3, 0.0)
    eff_w_sum = eff_w_h1 + eff_w_h3
    eff_w_h1 = np.where(eff_w_sum > 0, eff_w_h1 / eff_w_sum, np.nan)
    eff_w_h3 = np.where(eff_w_sum > 0, eff_w_h3 / eff_w_sum, np.nan)

    df['drift_blended'] = (
        eff_w_h1 * df['drift_h1n1'] +
        eff_w_h3 * df['drift_h3n2']
    )
    return df.dropna(subset=['drift_blended']).reset_index(drop=True)
