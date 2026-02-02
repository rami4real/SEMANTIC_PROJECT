import os
import csv
import json
import time
import sqlite3
import requests
import numpy as np
from urllib.parse import quote

# =========================
# 1) CONFIG
# =========================

JDM_URL = "https://jdm-api.demo.lirmm.fr/v0/"
CACHE_DB = "jdm_cache.sqlite"

# IMPORTANT:
# - Ces relations sont utilisées pour construire le BLOC 1 (A -> B).
# - Les labels (y) viennent de dataset_global.csv et peuvent avoir des noms différents.
# - Ici on aligne les noms avec ta liste de relations uniques.
RELATION_ID_MAP = {
    "r_depict": 172,
    "r_has_causitif": 42,       # (dans ton dataset: causitif)
    "r_has_property": 153,      # correspond à r_has_prop côté JDM
    "r_holo": 10,
    "r_lieu_origine": 15,       # on utilise r_lieu (id=15)
    "r_object_matière": 50,     # correspond à r_object>mater
    "r_own-1": 122,
    "r_processusagent": 137,    # correspond à r_processus>agent-1
    "r_processusinstr-1": 80,   # correspond à r_processus>instr
    "r_processuspatient": 138,  # correspond à r_processus>patient-1
    "r_product_of": 54,         # ID JDM = 54 :contentReference[oaicite:2]{index=2}
    "r_quantificateur": 174,    # correspond à r_quantificateur-1
    "r_social_tie": 113,        # correspond à r_has_social_tie_with
    "r_topic": 142,             # correspond à r_has_topic
}

R_ISA_ID = 6  # r_isa

HASH_DIM = 512
TOP_K_ISA = 10

# Split
TEST_RATIO = 0.2
SEED = 42

# Retry/backoff constant (plus rapide)
SLEEP_BETWEEN_RETRIES = 3  # seconds
MAX_RETRIES = 5
TIMEOUT_SECONDS = 3

# Checkpointing
SAVE_EVERY = 50
CHECKPOINT_TRAIN = "checkpoint_train.json"
CHECKPOINT_TEST = "checkpoint_test.json"
X_TRAIN_PATH = "X_train.npy"
Y_TRAIN_PATH = "y_train.npy"
X_TEST_PATH = "X_test.npy"
Y_TEST_PATH = "y_test.npy"

# Bad rows log
BAD_ROWS_CSV = "bad_rows.csv"

# Model persistence
MODEL_PROTOS_PATH = "model_prototypes.npy"
MODEL_LABELS_PATH = "model_labels.json"


# =========================
# 2) UTILITIES: checkpoint + bad row logging
# =========================

def load_checkpoint(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return int(obj.get("next_index", 0))
    except FileNotFoundError:
        return 0
    except Exception:
        return 0

def save_checkpoint(path: str, next_index: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"next_index": int(next_index)}, f)

def append_bad_row(row, reason: str) -> None:
    header = ["mot1", "mot2", "relation", "reason"]
    file_exists = True
    try:
        with open(BAD_ROWS_CSV, "r", encoding="utf-8") as _:
            pass
    except FileNotFoundError:
        file_exists = False

    with open(BAD_ROWS_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow([row[0], row[1], row[2], reason])


# =========================
# 3) API ERROR DETECTION
# =========================

def is_jdm_error(payload) -> bool:
    """
    Détecte une réponse API inutilisable.
    On reste volontairement conservateur.
    """
    if payload is None:
        return True
    if not isinstance(payload, dict):
        return True
    if payload.get("_error") is True:
        return True

    if "error" in payload and payload["error"]:
        return True
    if "message" in payload and isinstance(payload["message"], str) and "error" in payload["message"].lower():
        return True

    has_container = any(
        k in payload and isinstance(payload[k], list)
        for k in ["relations", "data", "edges", "result"]
    )
    if not has_container and not isinstance(payload.get("relation"), dict):
        return True

    return False


# =========================
# 4) CACHE SQLITE
# =========================

class SqliteCache:
    def __init__(self, path=CACHE_DB):
        self.conn = sqlite3.connect(path)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "k TEXT PRIMARY KEY, "
            "v TEXT NOT NULL, "
            "created_at INTEGER NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str):
        cur = self.conn.cursor()
        cur.execute("SELECT v FROM cache WHERE k=?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO cache(k, v, created_at) VALUES(?,?,?)",
            (key, json.dumps(value, ensure_ascii=False), int(time.time()))
        )
        self.conn.commit()


# =========================
# 5) CLIENT JDM (constant sleep + cache)
# =========================

class JDMClient:
    def __init__(self, cache: SqliteCache, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES):
        self.cache = cache
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    @staticmethod
    def _enc(term: str) -> str:
        return quote(term, safe="")

    def _get_json(self, url: str, cache_key: str):
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()

                if is_jdm_error(data):
                    raise ValueError("JDM payload indicates error/unexpected format")

                self.cache.set(cache_key, data)
                return data

            except Exception as e:
                if attempt == self.max_retries - 1:
                    data = {"_error": True, "_url": url, "_msg": str(e)}
                    self.cache.set(cache_key, data)
                    return data
                time.sleep(SLEEP_BETWEEN_RETRIES)

    def relation_from_to(self, node1: str, relation_id: int, node2: str):
        n1 = self._enc(node1)
        n2 = self._enc(node2)
        url = f"{JDM_URL}relations/from/{n1}/to/{n2}?types_ids={relation_id}"
        key = f"from_to|{node1}|{relation_id}|{node2}"
        return self._get_json(url, key)

    def relations_from(self, node: str, relation_id: int, min_weight: int = 1):
        n = self._enc(node)
        url = f"{JDM_URL}relations/from/{n}?types_ids={relation_id}&min_weight={min_weight}"
        key = f"from|{node}|{relation_id}|minw={min_weight}"
        return self._get_json(url, key)


# =========================
# 6) FEATURE EXTRACTION
# =========================

def extract_max_weight_from_to(resp_json) -> float:
    if not isinstance(resp_json, dict) or resp_json.get("_error"):
        return 0.0

    candidates = []
    for k in ["relations", "data", "edges", "result"]:
        if k in resp_json and isinstance(resp_json[k], list):
            candidates = resp_json[k]
            break

    if not candidates and isinstance(resp_json.get("relation"), dict):
        candidates = [resp_json["relation"]]

    best = 0.0
    for it in candidates:
        if not isinstance(it, dict):
            continue
        w = None
        for wk in ["w", "weight", "poids", "score"]:
            if wk in it:
                w = it[wk]
                break
        if w is None:
            continue
        try:
            w = float(w)
        except Exception:
            continue
        if w > best:
            best = w
    return best

def extract_isa_types(resp_json, top_k=TOP_K_ISA):
    if not isinstance(resp_json, dict) or resp_json.get("_error"):
        return []

    items = []
    for k in ["relations", "data", "edges", "result"]:
        if k in resp_json and isinstance(resp_json[k], list):
            items = resp_json[k]
            break

    pairs = []
    for it in items:
        if not isinstance(it, dict):
            continue

        target = None
        for tk in ["node2", "target", "to", "name2", "label2", "term2"]:
            if tk in it:
                target = it[tk]
                break
        if isinstance(target, dict):
            for nk in ["name", "label", "term"]:
                if nk in target:
                    target = target[nk]
                    break

        w = None
        for wk in ["w", "weight", "poids", "score"]:
            if wk in it:
                w = it[wk]
                break

        if target is None or w is None:
            continue

        try:
            w = float(w)
        except Exception:
            continue
        if w <= 0:
            continue

        pairs.append((str(target), w))

    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

def stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# =========================
# 7) VECTOR BUILDER
# =========================

class VectorBuilder:
    def __init__(self, jdm: JDMClient):
        self.jdm = jdm
        self.rel_names = list(RELATION_ID_MAP.keys())

    @property
    def dim(self):
        return 2 * len(self.rel_names) + HASH_DIM + 3

    def build_vector(self, A: str, B: str) -> np.ndarray:
        # BLOC 1: direct A->B
        bloc1 = []
        for rname in self.rel_names:
            rid = RELATION_ID_MAP[rname]
            resp = self.jdm.relation_from_to(A, rid, B)
            w = extract_max_weight_from_to(resp)
            exists = 1.0 if w > 0 else 0.0
            bloc1.extend([exists, w])
        bloc1 = np.array(bloc1, dtype=np.float32)

        # BLOC 2: r_isa of B
        bloc2 = np.zeros((HASH_DIM,), dtype=np.float32)
        isa_resp = self.jdm.relations_from(B, R_ISA_ID, min_weight=1)
        isa_pairs = extract_isa_types(isa_resp, top_k=TOP_K_ISA)
        for term, w in isa_pairs:
            idx = stable_hash(term.lower()) % HASH_DIM
            bloc2[idx] += float(w)

        # BLOC 3: stats
        if len(isa_pairs) == 0:
            isa_count, isa_max, isa_mean = 0.0, 0.0, 0.0
        else:
            ws = np.array([p[1] for p in isa_pairs], dtype=np.float32)
            isa_count = float(len(ws))
            isa_max = float(ws.max())
            isa_mean = float(ws.mean())
        bloc3 = np.array([isa_count, isa_max, isa_mean], dtype=np.float32)

        return np.concatenate([bloc1, bloc2, bloc3], axis=0)


# =========================
# 8) PROTOTYPE CLASSIFIER
# =========================

def l2_normalize_rows(X: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.sqrt((X * X).sum(axis=1, keepdims=True)) + eps
    return X / n

class PrototypeClassifier:
    def __init__(self):
        self.protos = None

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        X = X.astype(np.float32)
        y = y.astype(int)

        D = X.shape[1]
        self.protos = np.zeros((num_classes, D), dtype=np.float32)

        for c in range(num_classes):
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            self.protos[c] = X[idx].mean(axis=0)

        self.protos = l2_normalize_rows(self.protos)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = l2_normalize_rows(X.astype(np.float32))
        scores = Xn @ self.protos.T
        return scores.argmax(axis=1)


# =========================
# 9) METRICS
# =========================

def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def f1_macro(y_true, y_pred, num_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    f1s = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else 0.0


# =========================
# 10) DATASET LOAD + SPLIT (STRATIFIED + SHUFFLE)
# =========================

def load_dataset_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(f, dialect=dialect)

        for r in reader:
            a = (r.get("mot1") or "").strip()
            b = (r.get("mot2") or "").strip()
            rel = (r.get("relation") or r.get("label") or "").strip()
            if not a or not b or not rel:
                continue
            rows.append((a, b, rel))
    return rows

def make_label_map(rows):
    labels = sorted(list({rel for _, _, rel in rows}))
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label

def train_test_split(rows, test_ratio=TEST_RATIO, seed=SEED):
    """
    Stratified split + shuffle, pour éviter un test set dominé par un seul label.
    """
    rng = np.random.default_rng(seed)

    by_label = {}
    for row in rows:
        lab = row[2]
        by_label.setdefault(lab, []).append(row)

    train_rows = []
    test_rows = []

    for lab, items in by_label.items():
        items = items.copy()
        rng.shuffle(items)

        if len(items) <= 1:
            train_rows.extend(items)
            continue

        n_test = max(1, int(len(items) * test_ratio))
        test_rows.extend(items[:n_test])
        train_rows.extend(items[n_test:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


# =========================
# 11) RESUMABLE VECTORISATION
# =========================

def build_Xy_resumable(rows, vb: VectorBuilder, label2id: dict,
                       X_path: str, y_path: str, checkpoint_path: str):
    N = len(rows)
    D = vb.dim

    start_i = load_checkpoint(checkpoint_path)

    if start_i > 0:
        try:
            X = np.load(X_path)
            y = np.load(y_path)
            if X.shape != (N, D) or y.shape != (N,):
                X = np.zeros((N, D), dtype=np.float32)
                y = np.zeros((N,), dtype=np.int32)
                start_i = 0
        except Exception:
            X = np.zeros((N, D), dtype=np.float32)
            y = np.zeros((N,), dtype=np.int32)
            start_i = 0
    else:
        X = np.zeros((N, D), dtype=np.float32)
        y = np.zeros((N,), dtype=np.int32)

    print(f"Resume vectorization at index {start_i}/{N}")

    for i in range(start_i, N):
        a, b, rel = rows[i]

        try:
            vec = vb.build_vector(a, b)

            # Si tout est à zéro => très probablement "pas de data" ou erreur persistante
            if float(vec.sum()) == 0.0:
                append_bad_row((a, b, rel), reason="all_zero_vector_possible_api_failure")
                y[i] = -1
            else:
                X[i] = vec
                y[i] = label2id[rel]

        except Exception as e:
            append_bad_row((a, b, rel), reason=f"vector_build_exception: {e}")
            y[i] = -1

        if (i + 1) % SAVE_EVERY == 0:
            np.save(X_path, X)
            np.save(y_path, y)
            save_checkpoint(checkpoint_path, i + 1)
            print(f"Saved checkpoint at {i+1}/{N}")

    np.save(X_path, X)
    np.save(y_path, y)
    save_checkpoint(checkpoint_path, N)
    print("Vectorization complete and saved.")
    return X, y


# =========================
# 12) MAIN
# =========================

def main():
    csv_path = "dataset_50.csv"

    print("1) Chargement dataset...")
    rows = load_dataset_csv(csv_path)
    if len(rows) < 10:
        raise RuntimeError("Dataset trop petit ou colonnes incorrectes. Vérifiez mot1,mot2,relation.")
    print(f"   {len(rows)} lignes OK.")

    label2id, id2label = make_label_map(rows)
    C = len(label2id)
    print(f"2) Classes (relations) = {C}")
    print("   Exemple mapping:", list(label2id.items())[:10])

    train_rows, test_rows = train_test_split(rows)
    print(f"3) Split (stratified+shuffle): train={len(train_rows)} / test={len(test_rows)}")

    cache = SqliteCache(CACHE_DB)
    jdm = JDMClient(cache)
    vb = VectorBuilder(jdm)

    print(f"4) Dimension vecteur = {vb.dim}")

    # Vectorisation (reprend au checkpoint)
    print("5) Vectorisation TRAIN (resumable, avec cache JDM)...")
    X_train, y_train = build_Xy_resumable(
        train_rows, vb, label2id,
        X_path=X_TRAIN_PATH, y_path=Y_TRAIN_PATH,
        checkpoint_path=CHECKPOINT_TRAIN
    )

    print("6) Vectorisation TEST (resumable, avec cache JDM)...")
    X_test, y_test = build_Xy_resumable(
        test_rows, vb, label2id,
        X_path=X_TEST_PATH, y_path=Y_TEST_PATH,
        checkpoint_path=CHECKPOINT_TEST
    )

    # Remove bad rows (y == -1)
    train_mask = y_train >= 0
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    test_mask = y_test >= 0
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Train or load model
    model_exists = os.path.exists(MODEL_PROTOS_PATH) and os.path.exists(MODEL_LABELS_PATH)

    if model_exists:
        print("7) Modèle déjà sauvegardé -> chargement...")
        clf = PrototypeClassifier()
        clf.protos = np.load(MODEL_PROTOS_PATH)

        with open(MODEL_LABELS_PATH, "r", encoding="utf-8") as f:
            id2label_loaded = json.load(f)
        id2label_loaded = {int(k): v for k, v in id2label_loaded.items()}
        id2label = id2label_loaded
    else:
        print(f"7) Entraînement PrototypeClassifier... (train used={len(y_train)}, test used={len(y_test)})")
        clf = PrototypeClassifier()
        clf.fit(X_train, y_train, num_classes=C)

        print("   Sauvegarde du modèle...")
        np.save(MODEL_PROTOS_PATH, clf.protos)
        with open(MODEL_LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False, indent=2)

    print("8) Prédiction + évaluation...")
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    f1 = f1_macro(y_test, y_pred, num_classes=C)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1:.4f}")

    print("\nExemples (10) : vrai -> prédit")
    for i in range(min(10, len(y_test))):
        rel_true = id2label[int(y_test[i])]
        rel_pred = id2label[int(y_pred[i])]
        print(f"- {rel_true} -> {rel_pred}")

if __name__ == "__main__":
    main()
