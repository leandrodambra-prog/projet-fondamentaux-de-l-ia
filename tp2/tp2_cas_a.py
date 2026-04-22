import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap
import scipy.sparse as sp
import os

# Create directories for results if they don't exist
os.makedirs('resultats', exist_ok=True)

print("--- Étape 2 : Préparation des données ---")
# Chargement depuis URL publique — IBM Telco Customer Churn
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Nettoyage : TotalCharges contient des espaces
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encodage
df_enc = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)
X = df_enc.drop('Churn_Yes', axis=1)
y = df_enc['Churn_Yes'].astype(int)
feature_names = list(X.columns)

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train : {len(X_train)} | Test : {len(X_test)}")
print(f"Nombre de features : {len(feature_names)}")

print("\n--- Étape 3 : Modélisation ---")
modeles = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(
                                n_estimators=100, random_state=42,
                                eval_metric='logloss', verbosity=0
                           ),
}

resultats = {}
for nom, modele in modeles.items():
    modele.fit(X_train_sc, y_train)
    pred = modele.predict(X_test_sc)
    acc = accuracy_score(y_test, pred)
    f1_mac = f1_score(y_test, pred, average='macro')
    f1_wei = f1_score(y_test, pred, average='weighted')
    resultats[nom] = {'accuracy': acc, 'f1_macro': f1_mac, 'f1_weighted': f1_wei}
    print(f"{nom:22s} -> Accuracy : {acc*100:.1f}%  |  F1-macro : {f1_mac:.3f}")

# Visualisation Comparaison
df_resultats = pd.DataFrame(resultats).T
plt.figure(figsize=(9, 6))
df_resultats[['accuracy', 'f1_macro']].plot(kind='bar', ax=plt.gca())
plt.title('Comparaison des modèles (Cas A : Churn)')
plt.ylabel('Score')
plt.xticks(rotation=20)
plt.ylim(0.5, 1.0)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('resultats/comparaison_modeles.png')
plt.close()

print("\n--- Étape 4 : Évaluation approfondie (XGBoost) ---")
NOM_MEILLEUR = "XGBoost"
meilleur = modeles[NOM_MEILLEUR]
y_pred = meilleur.predict(X_test_sc)

print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de confusion — {NOM_MEILLEUR}')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('resultats/confusion_matrix.png')
plt.close()

# Évolution Random Forest
n_estimators_range = [10, 25, 50, 100, 200]
scores_rf = []
for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_temp.fit(X_train_sc, y_train)
    pred_temp = rf_temp.predict(X_test_sc)
    scores_rf.append(f1_score(y_test, pred_temp, average='macro'))

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, scores_rf, 'g-o')
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("F1-score macro")
plt.title("Évolution des performances — Random Forest")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultats/rf_evolution.png')
plt.close()

print("\n--- Étape 5 : Explicabilité SHAP ---")
rf_model = modeles["Random Forest"]
X_sample = X_test_sc[:200]
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

# Correction pour les versions de SHAP
if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

plt.figure()
shap.summary_plot(
    sv, X_sample,
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title("SHAP — Top 15 variables influentes (Churn=Yes)")
plt.tight_layout()
plt.savefig('resultats/shap_summary.png', bbox_inches='tight')
plt.close()

# Get Top 3 features from SHAP
mean_abs_shap = np.abs(sv).mean(axis=0)
top_3_indices = np.argsort(mean_abs_shap)[-3:][::-1]
top_3_vars = [feature_names[i] for i in top_3_indices]
print(f"Top 3 Variables : {top_3_vars}")

# Save final metrics for the slide
with open('resultats/metrics.txt', 'w') as f:
    f.write(f"Accuracy: {resultats[NOM_MEILLEUR]['accuracy']*100:.1f}\n")
    f.write(f"F1-macro: {resultats[NOM_MEILLEUR]['f1_macro']:.3f}\n")
    f.write(f"F1-weighted: {resultats[NOM_MEILLEUR]['f1_weighted']:.3f}\n")
    f.write(f"Top1: {top_3_vars[0]}\n")
    f.write(f"Top2: {top_3_vars[1]}\n")
    f.write(f"Top3: {top_3_vars[2]}\n")

print("\n--- Terminée ! Résultats sauvegardés dans le dossier 'resultats/' ---")
