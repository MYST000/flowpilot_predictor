"""Five estimators with shared support and nonnegative monotone quantiles."""
from collections import defaultdict, deque
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor
from predictor import QUANTILES, NAMES
from predictor.events import event_order
from predictor.features import extract, group_key, backend_key, argument_text

ALGORITHMS = ('empirical', 'ewma', 'cluster', 'qrf', 'lightgbm')


def monotone(values):
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('invalid quantiles')
    return np.sort(np.maximum(q, 0.0))


def weighted_quantiles(y, weights):
    y, weights = np.asarray(y, float), np.asarray(weights, float)
    if y.ndim != 1 or y.shape != weights.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('invalid weights')
    valid = weights > 0
    y, weights = y[valid], weights[valid]
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(weights).all():
        raise ValueError('invalid weighted distribution')
    order = np.argsort(y)
    y, weights = y[order], weights[order]
    cdf = np.cumsum(weights) / weights.sum()
    return y[np.minimum(np.searchsorted(cdf, np.asarray(QUANTILES) - 1e-12, side='left'), len(y)-1)]


class Base:
    def __init__(self, config):
        self.config = dict(config)
        self.state_version = 0

    def fit(self, samples):
        if not samples:
            raise ValueError('empty training data')
        self.samples = samples
        self.contexts = [s['context'] for s in samples]
        self.y = np.array([s['labels'][self.config['target']] for s in samples], float)
        if not np.isfinite(self.y).all() or (self.y < 0).any():
            raise ValueError('missing/invalid training labels')
        self.groups, self.backends = defaultdict(list), defaultdict(list)
        for i, c in enumerate(self.contexts):
            self.groups[group_key(c)].append(i)
            self.backends[backend_key(c)].append(i)
        return self

    def enough(self, indices):
        return len(indices) >= self.config['min_samples'] and len({self.samples[i]['task_group_id'] for i in indices}) >= self.config['min_groups']

    def context_error(self, c):
        if c.get('resolution') not in ('LOCAL_ONLY', 'LOCAL_LEADER'):
            return 'unsupported_resolution'
        if c.get('execution_mode') != 'serial':
            return 'unsupported_execution_mode'
        if group_key(c) not in self.groups:
            return 'unknown_backend_tool_or_version'
        return None

    def reference(self, c):
        error = self.context_error(c)
        if error:
            return [], error
        ids = self.groups[group_key(c)]
        if self.enough(ids):
            return ids, None
        backend = self.backends[backend_key(c)]
        if not self.enough(backend):
            return [], 'insufficient_backend_support'
        return backend, 'sparse_tool_backend_fallback'

    def result(self, q, indices, reason=None, method=None, effective_n=None):
        count = len(indices)
        group_count = len({self.samples[i]['task_group_id'] for i in indices})
        return {
            'duration_ms': None if q is None else dict(zip(NAMES, monotone(q).tolist())),
            'method': method or self.name,
            'online_state_version': self.state_version,
            'support': {'status': 'supported' if q is not None else 'unsupported',
                        'reference_rows': count, 'reference_task_groups': group_count,
                        'effective_rows': effective_n,
                        'q99_expected_tail_rows': count * .01,
                        'q99_low_support': count < 1000 or group_count < 30},
            'fallback': {'used': reason is not None, 'reason': reason},
        }

    def predict(self, c):
        ids, reason = self.reference(c)
        q = np.quantile(self.y[ids], QUANTILES) if ids else None
        return self.result(q, ids, reason)

    def remaining_from_dispatch(self, c, elapsed_ms):
        """Conditional empirical RTT, not batch-ready; fallback declared explicitly."""
        if not np.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError('invalid elapsed')
        ids, reason = self.reference(c)
        alive = [i for i in ids if self.y[i] > elapsed_ms]
        if ids and not self.enough(alive):
            alive = [i for i in self.backends[backend_key(c)] if self.y[i] > elapsed_ms]
            reason = 'sparse_survivors_backend_fallback'
        if not self.enough(alive):
            return self.result(None, alive, reason or 'insufficient_survivors', 'empirical_conditional')
        return self.result(np.quantile(self.y[alive] - elapsed_ms, QUANTILES), alive, reason, 'empirical_conditional')

    def feedback_snapshot(self, c):
        return None

    def observe(self, c, y, group, prediction_state=None):
        # Frozen estimators do not fit on evaluation labels.
        return None


class Empirical(Base):
    name = 'empirical'


class EWMA(Base):
    name = 'ewma'

    def fit(self, samples):
        super().fit(samples)
        self.means = {}
        self.residuals = {k: deque(maxlen=self.config['residual_buffer']) for k in self.groups}
        pending = {}
        # T1 snapshots must survive other calls returning before this call.
        for _, _, _, kind, _, i in event_order(samples):
            sample = samples[i]
            if kind == 'predict':
                pending[i] = self.feedback_snapshot(sample['context'])
            else:
                self.observe(sample['context'], sample['labels'][self.config['target']],
                             sample['task_group_id'], prediction_state=pending.pop(i))
        self.state_version = 0
        return self

    def feedback_snapshot(self, c):
        return {'key': group_key(c), 'log_center': self.means.get(group_key(c))}

    def observe(self, c, y, group, prediction_state=None):
        if not np.isfinite(y) or y < 0:
            raise ValueError('invalid feedback')
        if self.context_error(c):
            return
        k = group_key(c)
        if prediction_state is None or prediction_state['key'] != k:
            raise ValueError('EWMA feedback requires its prediction-time snapshot')
        z = np.log1p(y)
        center = prediction_state['log_center']
        if center is not None:
            self.residuals[k].append((float(z-center), group))
        self.means[k] = ((1-self.config['ewma_alpha']) * self.means[k] + self.config['ewma_alpha'] * z
                         if k in self.means else z)
        self.state_version += 1

    def predict(self, c):
        error = self.context_error(c)
        if error:
            return self.result(None, [], error)
        k = group_key(c)
        buffer = self.residuals[k]
        groups = {g for _, g in buffer}
        if len(buffer) < self.config['min_samples'] or len(groups) < self.config['min_groups']:
            ids, reason = self.reference(c)
            q = np.quantile(self.y[ids], QUANTILES) if ids else None
            return self.result(q, ids, reason or 'sparse_residuals_empirical_fallback', 'empirical')
        q = np.expm1(np.clip(self.means[k] + np.quantile([r for r, _ in buffer], QUANTILES), 0, 50))
        result = self.result(q, self.groups[k])
        result['support'].update(reference_rows=len(buffer), reference_task_groups=len(groups),
                                 fit_reference_rows=len(self.groups[k]),
                                 residual_rows=len(buffer), residual_task_groups=len(groups),
                                 q99_expected_tail_rows=len(buffer)*.01,
                                 q99_low_support=len(buffer) < 1000 or len(groups) < 30)
        return result


class Cluster(Base):
    name = 'cluster'

    def fit(self, samples):
        super().fit(samples)
        self.clusterers = {}
        self.members = {}
        for k, ids in self.groups.items():
            texts = [argument_text(self.contexts[i]) for i in ids]
            n = min(self.config['n_clusters'], len(ids)//self.config['min_samples'], len(set(texts)))
            if n < 2 or not self.enough(ids):
                continue
            vector = TfidfVectorizer(max_features=self.config['max_tfidf_features'], token_pattern=r'(?u)\b\w+\b')
            try:
                x = vector.fit_transform(texts)
            except ValueError as exc:
                if 'empty vocabulary' in str(exc):
                    continue
                raise
            km = KMeans(n_clusters=n, n_init=10, random_state=self.config['seed'])
            labels = km.fit_predict(x)
            self.clusterers[k] = vector, km
            for label in range(n):
                self.members[k, label] = [i for i, value in zip(ids, labels) if value == label]
        return self

    def predict(self, c):
        ids, reason = self.reference(c)
        if not ids:
            return self.result(None, [], reason)
        k = group_key(c)
        if k in self.clusterers:
            vector, km = self.clusterers[k]
            x = vector.transform([argument_text(c)])
            if x.nnz:
                cluster = int(km.predict(x)[0])
                members = self.members[k, cluster]
                if self.enough(members):
                    return self.result(np.quantile(self.y[members], QUANTILES), members)
                reason = 'sparse_cluster_empirical_fallback'
            else:
                reason = 'out_of_vocabulary_empirical_fallback'
        return self.result(np.quantile(self.y[ids], QUANTILES), ids, reason or 'no_cluster_empirical_fallback', 'empirical')


class Tabular(Base):
    def matrix(self, contexts, fit=False):
        features = [extract(c, self.config['feature_level']) for c in contexts]
        if fit:
            self.vector = DictVectorizer(sparse=False)
            return self.vector.fit_transform(features)
        return self.vector.transform(features)

    def transform_y(self):
        return np.log1p(self.y) if self.config['target_transform'] == 'log1p' else self.y


class QRF(Tabular):
    """Leaf-weighted empirical CDF over all fit responses (not tree-mean quantiles).

    Each tree routes all fit rows to leaves. Query weight for a row is the
    forest-average of 1 / leaf_population for trees sharing its leaf.
    """
    name = 'qrf'

    def fit(self, samples):
        super().fit(samples)
        x = self.matrix(self.contexts, fit=True)
        self.forest = RandomForestRegressor(n_estimators=self.config['n_estimators'],
            min_samples_leaf=self.config['min_samples_leaf'], max_depth=self.config['max_depth'],
            max_features=.8, n_jobs=self.config['threads'], random_state=self.config['seed'])
        self.forest.fit(x, self.transform_y())
        leaves = self.forest.apply(x)
        self.leaf_members = []
        for t in range(leaves.shape[1]):
            self.leaf_members.append({leaf: np.flatnonzero(leaves[:, t] == leaf) for leaf in np.unique(leaves[:, t])})
        return self

    def predict(self, c):
        ids, reason = self.reference(c)
        if not ids:
            return self.result(None, [], reason)
        # Shared trees learn across supported tools; report the actual CDF support.
        leaves = self.forest.apply(self.matrix([c]))[0]
        weights = np.zeros(len(self.y))
        for t, leaf in enumerate(leaves):
            members = self.leaf_members[t][leaf]
            weights[members] += 1.0 / (len(members) * len(leaves))
        support = np.flatnonzero(weights > 0).tolist()
        effective_n = float(1 / np.sum(weights ** 2))
        result = self.result(weighted_quantiles(self.y, weights), support, effective_n=effective_n)
        result['support']['q99_low_support'] |= effective_n < 1000
        return result


class LightGBM(Tabular):
    name = 'lightgbm'

    def fit(self, samples):
        super().fit(samples)
        x = self.matrix(self.contexts, fit=True)
        self.heads = []
        for q in QUANTILES:
            model = LGBMRegressor(objective='quantile', alpha=q, n_estimators=self.config['n_estimators'],
                learning_rate=self.config['learning_rate'], max_depth=self.config['max_depth'],
                num_leaves=min(31, 2**self.config['max_depth']), min_child_samples=self.config['min_samples_leaf'],
                n_jobs=self.config['threads'], random_state=self.config['seed'],
                device_type='cpu', deterministic=True, force_col_wise=True, verbosity=-1)
            model.fit(x, self.transform_y())
            self.heads.append(model)
        return self

    def predict(self, c):
        ids, reason = self.reference(c)
        if not ids:
            return self.result(None, [], reason)
        x = self.matrix([c])
        # Booster avoids sklearn's generated feature-name warnings on ndarray inference.
        q = np.array([m.booster_.predict(x, num_threads=self.config['threads'])[0] for m in self.heads])
        if self.config['target_transform'] == 'log1p':
            q = np.expm1(np.clip(q, 0, 50))
        return self.result(q, self.groups[group_key(c)])


def make_model(algorithm, config):
    return dict(empirical=Empirical, ewma=EWMA, cluster=Cluster, qrf=QRF, lightgbm=LightGBM)[algorithm](config)
