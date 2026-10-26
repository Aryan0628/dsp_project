"""Duplicate audit of the kaggle4 (SARTAJ) train/test split.

For every test slice: is it byte-identical (MD5) to a training file, and what is
its best normalised cross-correlation rho with any training slice (64x64 grey,
zero mean, unit L2 norm; the criterion of Vangala et al., arXiv:2610.00421)?
A slice with rho >= 0.98 counts as having a "twin" in training.

With --models it also re-runs the shipped detection and typing models on every
test slice and reports their scores separately for slices with and without a twin.

Run from the repo root:
    python group11_documents/supporting/duplicate_audit.py [--models]
"""
import argparse, collections, hashlib, json, os, sys
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, 'backend', 'data', 'kaggle4')
TAU = 0.98
CLASSES = {'no_tumor': 0, 'glioma_tumor': 1, 'meningioma_tumor': 2, 'pituitary_tumor': 3}


def files(split):
    base = os.path.join(DATA, split)
    return [(c, os.path.join(base, c, f)) for c in sorted(os.listdir(base))
            if os.path.isdir(os.path.join(base, c)) for f in sorted(os.listdir(os.path.join(base, c)))]


def vec(path):
    v = np.asarray(Image.open(path).convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float64).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', action='store_true', help='also re-run the shipped models')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'duplicate_audit_results.json'))
    a = ap.parse_args()

    tr, te = files('train'), files('test')
    T = np.stack([vec(p) for _, p in tr])
    hashes = {md5(p) for _, p in tr}
    rows = []
    for c, p in te:
        rho = float((T @ vec(p)).max())
        rows.append(dict(cls=c, y=CLASSES[c], path=os.path.basename(p), rho=rho, byte=md5(p) in hashes))

    print(f'test slices: {len(rows)}   byte-identical: {sum(r["byte"] for r in rows)}')
    for t in (0.90, 0.95, 0.98, 0.99, 0.995):
        print(f'  rho >= {t}: {sum(r["rho"] >= t for r in rows)}')
    for twin in (True, False):
        sub = [r for r in rows if (r['rho'] >= TAU) == twin]
        print(('twin' if twin else 'no twin'), len(sub), dict(collections.Counter(r['cls'] for r in sub)))

    if a.models:
        sys.path.insert(0, os.path.join(ROOT, 'backend', 'src'))
        from braintumor.train import load_model
        from braintumor.predict import analyse_image
        from braintumor.preprocess import load_image
        B, M = load_model(), load_model(multiclass=True)
        for r, (c, p) in zip(rows, te):
            img = load_image(p)
            r['pb'] = int(analyse_image(img, B)['label'])  # 1 = tumor
            r['pm'] = int(analyse_image(img, M)['label'])  # index into CLASSES
        for twin in (True, False):
            sub = [r for r in rows if (r['rho'] >= TAU) == twin]
            tum = [r for r in sub if r['y'] != 0]
            sens = sum(r['pb'] == 1 for r in tum) / len(tum) if tum else float('nan')
            acc = sum(r['pm'] == r['y'] for r in sub) / len(sub)
            print(f"{'twin' if twin else 'no twin'}: detection sensitivity {sens:.3f}, typing accuracy {acc:.3f}")

    json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
