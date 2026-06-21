# ⚡ Détection d'Anomalies de Consommation Électrique par BiLSTM-Autoencoder

**M1 GSIT – Mme Amel KHEITER · Année Universitaire 2025/2026**

---

## 📋 Description

Système complet de détection d'anomalies dans la consommation électrique résidentielle
en utilisant un **BiLSTM Autoencoder** entraîné exclusivement sur des patterns normaux.

Les anomalies sont détectées par **erreur de reconstruction élevée** : si le modèle ne
sait pas reconstruire une séquence, c'est qu'elle est anormale.

---

## 🏗️ Architecture BiLSTM Autoencoder

```
Input  (batch, 24 heures, 4 features)
   │
   ▼
Bidirectional LSTM (64 units)   ← Encoder
   │
   ▼
Dense Latent Space (16 units)
   │
   ▼
RepeatVector (24)
   │
   ▼
LSTM Decoder (64 units)
   │
   ▼
TimeDistributed Dense (4 features)
   │
   ▼
Output (batch, 24 heures, 4 features)
```

---

## 📁 Structure du Projet

```
anomaly_detection/
├── app.py              ← Dashboard Streamlit (interface web)
├── main.py             ← Pipeline complet (train + évaluation)
├── model.py            ← Architectures : BiLSTM AE, Simple AE, IsolationForest
├── preprocessing.py    ← Chargement, nettoyage, fenêtrage glissant
├── utils.py            ← Métriques, courbes ROC, visualisations
├── requirements.txt    ← Dépendances Python
├── data/               ← Dataset UCI (téléchargé automatiquement)
├── models/             ← Modèles .keras entraînés
├── plots/              ← Figures PNG générées
└── outputs/            ← Rapports CSV
```

---

## ⚙️ Installation

### 1. Cloner / copier le projet

```bash
cd anomaly_detection
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Option A – Pipeline complet (entraînement + évaluation)

```bash
python main.py
```

Ce que ça fait :
1. Télécharge le dataset UCI automatiquement (~120 MB)
2. Nettoie, resample (horaire), normalise les données
3. Injecte des anomalies synthétiques (spikes + drops)
4. Entraîne le **BiLSTM Autoencoder**
5. Entraîne le **Simple Autoencoder** (baseline)
6. Entraîne l'**Isolation Forest** (baseline)
7. Génère tous les plots dans `plots/`
8. Affiche le tableau comparatif des métriques
9. Sauvegarde les modèles dans `models/`

Durée estimée : **5–15 minutes** selon le hardware.

---

### Option B – Interface Streamlit

```bash
# Lancer main.py d'abord pour entraîner les modèles
python main.py

# Puis lancer l'interface web
streamlit run app.py
```

Ouvre automatiquement sur http://localhost:8501

**Fonctionnalités :**
- Upload d'un fichier CSV personnalisé
- Choix du modèle (BiLSTM / Simple AE / Isolation Forest)
- Réglage du seuil de détection (slider)
- Graphes interactifs (Plotly)
- Export du rapport d'anomalies en CSV

---

## 📊 Dataset : UCI Household Power Consumption

| Propriété     | Valeur |
|---------------|--------|
| Source        | UCI Machine Learning Repository |
| Enregistrements | ~2 millions (minutes, 2006–2010) |
| Après resample | ~35 000 (horaire) |
| Téléchargement | Automatique via `preprocessing.py` |
| Taille archive | ~20 MB (zip) |

**Features utilisées :**

| Feature | Description | Unité |
|---------|-------------|-------|
| `Global_active_power` | Puissance active globale | kW |
| `Global_reactive_power` | Puissance réactive globale | kW |
| `Voltage` | Tension moyenne | V |
| `Global_intensity` | Intensité moyenne | A |

---

## 🔍 Pipeline de Détection

```
Raw Data (minute)
      │
      ▼
Nettoyage (NaN → interpolation)
      │
      ▼
Resample horaire (moyenne)
      │
      ▼
Sélection de 4 features
      │
      ▼
Normalisation MinMaxScaler (fit sur train seulement)
      │
      ▼
Fenêtres glissantes 24h → [N, 24, 4]
      │
      ├──────────────────────────────────┐
      │  Injection anomalies synthétiques│
      │  (spikes ×3.5 + drops ×0.1)     │
      └──────────────────────────────────┘
      │
      ▼
Entraînement sur données NORMALES uniquement
      │
      ▼
Erreur de reconstruction MSE par fenêtre
      │
      ▼
Seuil = 95e percentile des erreurs d'entraînement
      │
      ▼
Classification : erreur > seuil → ANOMALIE
```

---

## 📈 Résultats Attendus

| Modèle | F1 Score (typ.) | AUC (typ.) |
|--------|-----------------|------------|
| BiLSTM Autoencoder | 0.75 – 0.90 | 0.85 – 0.95 |
| Simple Autoencoder | 0.65 – 0.80 | 0.78 – 0.88 |
| Isolation Forest   | 0.60 – 0.75 | 0.72 – 0.85 |

*Résultats variables selon la durée d'entraînement et le seuil choisi.*

---

## 🖼️ Plots Générés

| Fichier | Description |
|---------|-------------|
| `plots/consumption_overview.png` | Aperçu des 4 features sur toute la période |
| `plots/feature_correlation.png` | Matrice de corrélation |
| `plots/training_history_BiLSTM_AE.png` | Courbes train/val loss |
| `plots/error_histogram_BiLSTM_AE.png` | Distribution des erreurs + seuil |
| `plots/anomaly_timeline_BiLSTM_AE.png` | Consommation + anomalies détectées |
| `plots/confusion_matrix_BiLSTM_AE.png` | Matrice de confusion |
| `plots/roc_curves_comparison.png` | Courbes ROC des 3 modèles |
| `plots/metrics_comparison.png` | Bar chart comparatif F1/Precision/Recall/AUC |

---

## 🧪 Anomalies Synthétiques

Le projet injecte deux types d'anomalies dans les données de test :

- **Spikes** : multiplication de `Global_active_power` par **3.5×** (surconsommation)
- **Drops** : réduction de toutes les features à **10%** de leur valeur (sous-consommation / coupure)

Proportion : **5% des échantillons** de test → labels binaires pour évaluation supervisée.

---

## 📦 Dépendances Principales

```
tensorflow >= 2.10
scikit-learn >= 1.1
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
streamlit >= 1.20
plotly >= 5.13
```

---

## 💡 Conseils d'Optimisation

1. **Augmenter les epochs** : changer `EPOCHS = 50` dans `main.py` pour de meilleurs résultats
2. **Ajuster le seuil** : utiliser le slider Streamlit (80–99e percentile)
3. **GPU** : TensorFlow détecte automatiquement un GPU NVIDIA → entraînement 5–10× plus rapide
4. **Mémoire** : réduire `BATCH_SIZE` à 32 si vous avez moins de 8 GB RAM

---

## 👨‍🎓 Informations Académiques

- **Module** : Machine Learning & Deep Learning
- **Niveau** : Master 1 – Gestion des Systèmes Industriels et Tertiaires (GSIT)
- **Rendu** : Code + Rapport (12p) + Présentation
