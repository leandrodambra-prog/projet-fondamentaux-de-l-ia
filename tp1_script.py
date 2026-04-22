import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Étape 1
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("=== EXPLORATION DES DONNÉES ===")
print(f"Forme du dataset : {df.shape}")
print(f"\nDistribution des classes :")
print(df['species'].value_counts())
print(f"\nValeurs manquantes : {df.isnull().sum().sum()}")
print(f"\nAperçu :")
print(df.head(3))
print(f"\nStatistiques descriptives :")
print(df.describe())

sns.pairplot(df, hue='species', markers=["o", "s", "D"], plot_kws=dict(alpha=0.7), diag_kind='hist')
plt.suptitle("Dataset Iris — Séparabilité des classes", y=1.02)
plt.savefig('iris_pairplot.png', bbox_inches='tight')

# Étape 2
le = LabelEncoder()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = le.fit_transform(df['species'])
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nEntraînement : {len(X_train)} échantillons | Test : {len(X_test)} échantillons")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_sc, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_sc, y_train)

print("Modèles entraînés.")

# Étape 3
y_pred_dt = dt.predict(X_test_sc)
y_pred_rf = rf.predict(X_test_sc)

print("\n=== COMPARAISON BASELINE vs RANDOM FOREST ===")
for nom, pred in [("Decision Tree (baseline)", y_pred_dt), ("Random Forest ", y_pred_rf)]:
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    print(f"{nom} -> Accuracy : {acc*100:.1f}% | F1-score (weighted) : {f1:.3f}")

print("\n=== RAPPORT DÉTAILLÉ — RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de confusion — Random Forest')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# Étape 4
print("\n=== TUNING & OVERFITTING ===")
n_estimators_range = [10, 25, 50, 100, 200, 500]
scores_cv = []
for n in n_estimators_range:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    cv = cross_val_score(m, X_train_sc, y_train, cv=5, scoring='accuracy')
    scores_cv.append(cv.mean())
    print(f"n_estimators={n:4d} -> CV accuracy : {cv.mean()*100:.1f}% (±{cv.std()*100:.1f}%)")

profondeurs = range(1, 20)
scores_train = []
scores_test = []

for d in profondeurs:
    m = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42)
    m.fit(X_train_sc, y_train)
    scores_train.append(m.score(X_train_sc, y_train))
    scores_test.append(m.score(X_test_sc, y_test))

plt.figure(figsize=(9, 4))
plt.plot(profondeurs, scores_train, 'b-o', label='Score entraînement')
plt.plot(profondeurs, scores_test, 'r-o', label='Score test')
plt.xlabel('Profondeur maximale (max_depth)')
plt.ylabel('Accuracy')
plt.title('Underfitting vs Overfitting — Random Forest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('overfitting.png')

# Étape 5
print("\n=== EXPLICABILITÉ SHAP ===")
rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_final.fit(X_train_sc, y_train)

explainer = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_test_sc)

plt.figure()
shap.summary_plot(
    shap_values, X_test_sc,
    feature_names=feature_names,
    class_names=list(le.classes_),
    plot_type='bar',
    show=False
)
plt.title("SHAP — Importance globale des variables (3 classes)")
plt.tight_layout()
plt.savefig('shap_summary.png')

importances_sklearn = rf_final.feature_importances_
if isinstance(shap_values, list):
    importances_shap = np.abs(shap_values[2]).mean(axis=0) # index 2 = virginica
else:
    importances_shap = np.abs(shap_values[:, :, 2]).mean(axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.barh(feature_names, importances_sklearn, color='steelblue')
ax1.set_title('Feature Importance — sklearn (MDI)')
ax1.set_xlabel('Importance')

ax2.barh(feature_names, importances_shap, color='darkorange')
ax2.set_title('Feature Importance — SHAP (classe virginica)')
ax2.set_xlabel('|SHAP value| moyen')

plt.tight_layout()
plt.savefig('feature_importance.png')

print("Terminé ! Toutes les images ont été générées.")
