# train_both_v5.py
# pip install numpy scikit-learn joblib torch torchvision matplotlib
import numpy as np, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA = 'dataset_worldcover_v5.npz'
BATCH = 64
EPOCHS = 20
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

d = np.load(DATA, allow_pickle=True)
CODES = d['CLASS_VALUES']; K = len(CODES)
LABELS = np.arange(K)
NAMES = [str(int(v)) for v in CODES]

# ---------- KNN ----------
Xtr,ytr = d['Xtr_knn'], d['ytr_knn']
Xva,yva = d['Xva_knn'], d['yva_knn']
Xte,yte = d['Xte_knn'], d['yte_knn']

knn = Pipeline([('scaler', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1))])
knn.fit(Xtr, ytr)

def report_knn(X, y, title, out_png):
    if X.size == 0: print(f'{title}: empty'); return
    pred = knn.predict(X)
    print(title)
    print(classification_report(y, pred, labels=LABELS, target_names=NAMES, digits=3, zero_division=0))
    cm = confusion_matrix(y, pred, labels=LABELS)
    plt.figure(figsize=(6,6)); plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.xlabel('Pred'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

report_knn(Xva, yva, 'KNN VAL',  'knn_val_cm.png')
report_knn(Xte, yte, 'KNN TEST', 'knn_test_cm.png')
dump(knn, 'knn_worldcover.joblib')

# ---------- CNN ----------
class PSet(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]).float(), torch.tensor(int(self.y[i]))

Xtr_c, ytr_c = d['Xtr_cnn'], d['ytr_cnn']
Xva_c, yva_c = d['Xva_cnn'], d['yva_cnn']
train_dl = DataLoader(PSet(Xtr_c, ytr_c), batch_size=BATCH, shuffle=True,  num_workers=0)
val_dl   = DataLoader(PSet(Xva_c, yva_c), batch_size=BATCH, shuffle=False, num_workers=0)

class SimpleCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6,32,3,padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(256, n)
    def forward(self, x): return self.fc(self.net(x).flatten(1))

model = SimpleCNN(K).to(DEVICE)

# robust class weights
cnt = np.bincount(ytr_c, minlength=K).astype(np.float32)
w = np.zeros(K, dtype=np.float32)
mask = cnt > 0
w[mask] = cnt[mask].sum() / (cnt[mask] * mask.sum())
crit = nn.CrossEntropyLoss(weight=torch.tensor(w).to(DEVICE))

opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

def run_epoch(dl, train):
    if len(dl.dataset)==0: return 0.0, 0.0
    model.train(train); tot, correct, n = 0.0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.set_grad_enabled(train):
            out = model(xb); loss = crit(out, yb)
        if train: opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*xb.size(0); correct += (out.argmax(1)==yb).sum().item(); n += xb.size(0)
    return tot/max(n,1), correct/max(n,1)

best, bad, patience = 0.0, 0, 5
for e in range(1, EPOCHS+1):
    tl, ta = run_epoch(train_dl, True)
    vl, va = run_epoch(val_dl,   False)
    sched.step(vl)
    print(f'E{e:02d}  train {ta:.3f}  val {va:.3f}')
    if va>best: best, bad = va, 0; torch.save(model.state_dict(),'cnn_worldcover.pt')
    else: bad += 1
    if bad>=patience: break

# Validation report
if len(val_dl.dataset)>0:
    model.load_state_dict(torch.load('cnn_worldcover.pt', map_location=DEVICE)); model.eval()
    P, T = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            P.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy()); T.append(yb.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    print('CNN VAL')
    print(classification_report(T, P, labels=LABELS, target_names=NAMES, digits=3, zero_division=0))
    cm = confusion_matrix(T, P, labels=LABELS)
    plt.figure(figsize=(6,6)); plt.imshow(cm, interpolation='nearest')
    plt.title('CNN VAL'); plt.xlabel('Pred'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig('cnn_val_cm.png', dpi=150); plt.close()

print('Saved: knn_worldcover.joblib, cnn_worldcover.pt, *_cm.png')
