"""write_21_final_candidate.py -- Task 2: write submissions/21_final.csv as bundle candidate.

Uses the cached 21_final OOF and final-model artefacts:
  - 5-seed stage-2 (scratch/21_cache/21_oof_seed{0..4}.npy)
  - trajectory blend at alpha=0.2 (scratch/19_cache/19_traj_final_preds.npy)
  - best smoothing: deg=1, lam=0.0, w=0.0 (no smoothing was best in 21_final)

Sanity-checks the output and prints CANDIDATE notice.
Does NOT push to hub as the primary 21_final skore report (that requires
the full skore.evaluate wrapper and is slow); instead puts a metadata dict.

PYTHONUTF8=1. ASCII-only prints.
"""
"""
PIPELINE FINAL - PRÉDICTION DU SCORE "TRUE OFF" (PARKINSON)


1. ARCHITECTURE ET CHEMINEMENT DU MODÈLE

L'objectif est de prédire le score moteur "vrai OFF" (sans effet médicamenteux), 
souvent manquant ou biaisé. Le pipeline final combine plusieurs approches :
  - Stage 1 : Un modèle de base (HGBR) lissé temporellement par patient.
  - Stage 2 : Un ensemble de 5 modèles HGBR (seeds multiples) entraînés sur 
    des features enrichies (InputCurveTransformer), capturant la dynamique 
    quadratique des états ON et OFF de chaque patient.
  - Modèle de Trajectoire : Un modèle dédié prédisant l'évolution quadratique 
    long terme de la maladie pour un patient donné.
  - Blend Final : Mélange des prédictions du Stage 2 (80%) et du modèle de 
    trajectoire (20%, alpha=0.2).

2. FEATURE ENGINEERING ET GESTION DES VALEURS NULLES (NaNs)

Face au fort taux de données manquantes (ex: 42% de scores OFF manquants) :
  - Gestion native : Le modèle HGBR utilise l'absence de donnée comme une 
    information prédictive à part entière.
  - Imputation intelligente ("off_hat") : Un modèle Ridge déduit les scores 
    OFF manquants à partir du score ON, de l'âge et de la dose LEDD.
  - Flags de missingness : Création de variables binaires et de fréquences 
    (ex: on_miss_frac) indiquant l'absence de certaines mesures.
  - Fallbacks : Utilisation de moyennes/médianes globales pour imputer les 
    données des nouveaux patients du set de test (cold-start).


"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials

# COMMENTAIRE : Importation des fonctions clés développées dans les étapes précédentes.
# ridge_poly_smooth effectue le lissage temporel intra-patient.
# InputCurveTransformer (de l'étape 17) gère les courbes quadratiques des inputs (ON/OFF) et l'imputation (off_hat).
from shared import ridge_poly_smooth, sanity_check
import step2_17_stage2_feats as _s17
InputCurveTransformer = _s17.InputCurveTransformer
build_s2_features     = _s17.build_s2_features

# Définition des chemins vers les données et les caches (pour éviter de tout recalculer)
DATA_DIR = PROJECT_ROOT / "data"
SUBS_DIR = PROJECT_ROOT / "submissions"
C11  = PROJECT_ROOT / "scratch" / "13_cache"  # Contient l'OOF du Stage 1 (11_tuned)
C17  = PROJECT_ROOT / "scratch" / "17_cache"  # Contient les features Stage 2 (Courbes)
C19  = PROJECT_ROOT / "scratch" / "19_cache"  # Contient les prédictions du modèle de trajectoire
C21  = PROJECT_ROOT / "scratch" / "21_cache"  # Cache pour ce script final
SUBS_DIR.mkdir(exist_ok=True)

# Configuration retenue pour le lissage du Stage 1 (de l'exp. 14)
CFG14    = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
# Hyperparamètres optimisés pour le HGBR du Stage 2
S2_PARAMS = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                 min_samples_leaf=50, l2_regularization=0.0)
SEEDS    = [0, 1, 2, 3, 4] # Pour l'ensembling (moyenne de 5 modèles)

# Résultats de la run 21_final
BEST_FOLDS_21 = None   # Sera recalculé plus bas
# La configuration finale de lissage (w_blend=0.0 signifie qu'on ne lisse pas la prédiction finale du test)
BEST_CFG_21   = dict(degree=1, ridge_lam=0.0, w_blend=0.0, use_hetero=False)
BEST_ALPHA    = 0.2    # Poids du modèle de trajectoire (20% Trajectoire, 80% Stage 2)
PASS_A_RMSE   = 3.9832
PASS_B_RMSE   = 3.9743

t0 = time.time()

# Chargement des datasets
X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
ages_a = X["age"].values
pids_a = X["patient_id"].values
y_vals = y.values.astype(float)

# Création des splits de validation croisée.
# GroupKFold garantit qu'un patient de validation n'est JAMAIS vu dans l'entraînement.
from sklearn.model_selection import GroupKFold
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

# ── Load cached 21_final OOF ──────────────────────────────────────────────────
print("Loading 21_final cached OOF seeds...")
oof21_seeds = []
# COMMENTAIRE : On charge les prédictions "Out-Of-Fold" (OOF) pré-calculées pour chacun des 5 modèles.
for seed in SEEDS:
    p = C21 / f"21_oof_seed{seed}.npy"
    if p.exists():
        oof21_seeds.append(np.load(p))
        print(f"  seed {seed}: loaded")
    else:
        print(f"  seed {seed}: MISSING - cannot write 21_final.csv without full OOF")
        sys.exit(1)

# On fait la moyenne des 5 modèles pour obtenir l'OOF final du Stage 2
oof21_avg = np.mean(oof21_seeds, axis=0)

# Apply trajectory blend
# COMMENTAIRE : Le modèle "Trajectoire" a été entrainé à deviner la courbe d'évolution.
# On le "blend" (mélange) avec les prédictions du modèle ensembliste.
traj_oof_path = C19 / "19_traj_oof.npy"
if traj_oof_path.exists():
    traj_oof = np.load(traj_oof_path)
    oof21_blended = (1 - BEST_ALPHA) * oof21_avg + BEST_ALPHA * traj_oof
    print(f"Traj blend applied (alpha={BEST_ALPHA}): "
          f"RMSE={np.sqrt(np.mean((oof21_blended-y_vals)**2)):.4f}")
else:
    oof21_blended = oof21_avg
    print("No traj OOF found; using 21_avg only")

# Recompute per-fold metrics with best cfg (w=0.0 means no smoothing)
# COMMENTAIRE : Recalcul de l'erreur par fold sur les données entraînées (Cross Validation RMSE).
folds_21 = []
for _, val_idx in cv_splits:
    sm = ridge_poly_smooth(oof21_blended[val_idx], ages_a[val_idx],
                           pids_a[val_idx],
                           BEST_CFG_21["w_blend"], BEST_CFG_21["degree"],
                           BEST_CFG_21["ridge_lam"])
    folds_21.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))

print(f"\n21_final OOF (Pass A):  {np.mean(folds_21):.4f} +/- {np.std(folds_21):.4f}")
print(f"Per-fold: {[round(f,4) for f in folds_21]}")
print(f"Pass B (shuffled):      {PASS_B_RMSE:.4f}")
print(f"Mean gain vs 17:        {3.9949 - np.mean(folds_21):.4f}")

# ── Build final test predictions ──────────────────────────────────────────────
print("\nBuilding test predictions...")

# COMMENTAIRE : On récupère les composants fondamentaux : 
# 1. Le transformateur de courbes (qui contient l'imputateur off_hat).
# 2. Les prédictions du Stage 1 (pour l'OOF et le Test).
ct_final    = joblib.load(C17 / "17_ct_final.joblib")
oof_11      = np.load(C11 / "11_oof_raw.npy")
raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")

# COMMENTAIRE : On lisse le Stage 1. Le Stage 2 utilisera ces prédictions lissées comme caractéristiques.
sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                X_test["patient_id"].values,
                                CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

# COMMENTAIRE : Construction des features du Stage 2. 
# C'est ici que 'build_s2_features' appelle 'InputCurveTransformer' qui applique 
# le Ridge pour imputer les 'off' manquants et construit les features d'agrégation.
X_train_s2 = build_s2_features(X.copy(), sm_train_s1, ct_final)
X_test_s2  = build_s2_features(X_test.copy(), sm_test_s1, ct_final)

# COMMENTAIRE : Inférence (Prédiction) avec nos 5 modèles HGBR (un par seed)
test_preds = []
for seed in SEEDS:
    mc = C21 / f"21_s2_final_seed{seed}.joblib"
    tc = C21 / f"21_raw_test_seed{seed}.npy"
    if mc.exists() and tc.exists():
        test_preds.append(np.load(tc))
        print(f"  seed {seed}: loaded from cache")
    else:
        print(f"  seed {seed}: fitting final model...")
        m2 = HistGradientBoostingRegressor(
            categorical_features="from_dtype", early_stopping=True,
            n_iter_no_change=20, random_state=seed, **S2_PARAMS)
        m2.fit(X_train_s2, y)
        raw_t = m2.predict(X_test_s2)
        joblib.dump(m2, mc)
        np.save(tc, raw_t)
        test_preds.append(raw_t)

from sklearn.ensemble import HistGradientBoostingRegressor

# Moyenne des 5 modèles sur le Test Set
raw_test_avg = np.mean(test_preds, axis=0)

# Apply trajectory blend to test
# COMMENTAIRE : Mélange final des prédictions Test avec le modèle de Trajectoire (Alpha = 0.2)
traj_test_path = C19 / "19_traj_final_preds.npy"
if traj_test_path.exists():
    traj_test = np.load(traj_test_path)
    raw_test_blended = (1 - BEST_ALPHA) * raw_test_avg + BEST_ALPHA * traj_test
    print(f"Traj blend applied to test (alpha={BEST_ALPHA})")
else:
    raw_test_blended = raw_test_avg
    print("No traj test preds found; using 21 only")

# No post-smoothing (w=0.0 was best)
# COMMENTAIRE : Borne finale imposée par l'UPDRS (0 = Normal, 132 = Sévérité maximale)
final_test = np.clip(raw_test_blended, 0, 132)

sub = pd.DataFrame({"target": final_test}, index=X_test.index)
sub.index.name = "Index"

# Vérifications de cohérence (pas de NaNs introduits, bon format) avant de sauvegarder.
sanity_check(sub, X_test, "21_final")

out_path = SUBS_DIR / "21_final.csv"
sub.to_csv(out_path)
print(f"\nWritten: {out_path}")

# ── Hub metadata put ──────────────────────────────────────────────────────────
# COMMENTAIRE : Envoi des métadonnées du modèle vers la plateforme Skore Hub pour le monitoring
hub_url = "PENDING"
try:
    import skore
    from skore import login
    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    meta = {
        "key": "21_final",
        "note": "CANDIDATE - 5-seed s2 + traj blend alpha=0.2, no post-smooth",
        "cv_rmse_pass_a": float(np.mean(folds_21)),
        "cv_rmse_pass_a_std": float(np.std(folds_21)),
        "cv_rmse_pass_b": float(PASS_B_RMSE),
        "folds_pass_a": [float(f) for f in folds_21],
        "mean_gain_vs_17": float(3.9949 - np.mean(folds_21)),
        "folds_improved_vs_17": int(sum(
            r17 > r21 for r17, r21 in
            zip([4.2602, 4.0026, 3.9085, 3.8914, 3.9119], folds_21))),
    }
    project.put("21_final_meta", meta)
    hub_url = (f"https://skore.probabl.ai/{cfg['workspace']}/"
               f"bobathon-esilv/cross-validations")
    print(f"Hub metadata pushed.")
except Exception as e:
    print(f"HUB PUT PENDING: ({e})")

# Résumé imprimé dans la console
print("\n" + "="*60)
print("CANDIDATE (DO NOT SUBMIT YET): submissions/21_final.csv")
print(f"  CV RMSE Pass A: {np.mean(folds_21):.4f} +/- {np.std(folds_21):.4f}")
print(f"  CV RMSE Pass B: {PASS_B_RMSE:.4f}")
print(f"  Mean gain vs 17: {3.9949 - np.mean(folds_21):.4f}  (threshold: 0.03)")
print(f"  Folds improved: {sum(r17>r21 for r17,r21 in zip([4.2602,4.0026,3.9085,3.8914,3.9119],folds_21))}/5")
print(f"  Hub URL: {hub_url}")
print("="*60)
print(f"\nTotal elapsed: {time.time()-t0:.0f}s")