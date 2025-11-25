# 📁 Splits 5-Fold Stratifiés Multi-Label

Ce dossier contient les fichiers associés à la construction des **splits 5-fold stratifiés multi-label**, utilisés dans le cadre de l’entraînement et de la validation du modèle de classification de défauts vidéos (projet QV/VideoPipe).

Cette étape assure une séparation robuste et équilibrée des données, alignée avec les pratiques des meilleures solutions de compétitions similaires.

---

## ## Étapes du projet

### **Étape 1-3: Création des splits 5-fold**

- **Objectif :**  
  Générer cinq sous-ensembles équilibrés du jeu de données, afin de permettre une validation croisée fiable pour un problème de classification multi-label.

- **Méthodes utilisées :**  
  - **Stratification multi-label** à l’aide de l’algorithme `MultilabelStratifiedKFold` (librairie `iterstrat`)  
  - Répartition équilibrée des classes dans chaque fold  
  - Propagation des folds au niveau des frames extraites  
  - Export des tables nécessaires à l’entraînement et à l’analyse

---

## ## Concept de "fold" et principe de la validation croisée

Un *fold* représente une partition du dataset.

Dans le cadre d’une **validation croisée en 5 parties (5-fold cross-validation)** :

1. Le dataset est divisé en cinq ensembles distincts.  
2. Pour chaque exécution d'entraînement :  
   - Quatre folds servent à l’apprentissage.  
   - Un fold sert à la validation.  
3. L’opération est répétée cinq fois, en faisant tourner le fold de validation.  
4. Cinq modèles indépendants sont alors obtenus.  

Cette méthode permet une évaluation plus stable, réduit la variance et améliore la robustesse des performances finales.

---

## ## Justification de l’utilisation d’une stratification *multi-label*

Dans le jeu de données QV :

- Une vidéo peut appartenir à plusieurs classes simultanément.  
- Les classes sont fortement déséquilibrées (certaines rares, d’autres majoritaires).  

Une stratification classique ne peut pas garantir un équilibre correct pour ce type de tâche.  
L’algorithme **MultilabelStratifiedKFold** permet de :

- conserver les proportions de chaque classe dans chaque fold,  
- traiter les cas multi-labels,  
- assurer une distribution homogène même pour les classes peu représentées,  
- éviter les biais dans l’évaluation du modèle.

Le tableau de distribution généré (voir section « Analyse ») confirme cet équilibre.

---

## ## Fichiers générés dans ce dossier

### **1. `video_folds_5fold.csv`**

Contient les informations relatives à chaque vidéo annotée.

| Colonne | Description |
|---------|-------------|
| `video_id` | Nom de la vidéo (ex. `d16427.mp4`) |
| `labels_list` | Liste brute des labels associés |
| `labels_str` | Version concaténée des labels (format texte) |
| `fold` | Numéro du fold (0 à 4) |
| `video_stem` | Identifiant de vidéo sans extension |

Ce fichier sert de base pour la création des datasets vidéo et la configuration des splits d'entraînement.

---

### **2. `frames_5_forstep1and2_folds.csv`**

Contient les informations pour chaque frame extraite.

| Colonne | Description |
|---------|-------------|
| `frame_path` | Chemin complet de la frame |
| `video_stem` | Identifiant de la vidéo associée |
| `labels_str` | Labels hérités de la vidéo |
| `fold` | Numéro du fold correspondant |

Toutes les frames issues d’une même vidéo héritent du même fold, afin de garantir une séparation stricte entre apprentissage et validation et d’éviter toute fuite de données.

---

## ## Analyse de la distribution des classes

Un fichier récapitulatif est généré pour documenter l'équilibre de la stratification :

```text
reports/tables/preprocessing/class_distribution_per_fold.csv
```

Ce tableau présente, pour chaque classe :

- le nombre de vidéos présentes dans chacun des cinq folds,  
- le total global de vidéos par classe.

Les résultats montrent une distribution homogène, démontrant le bon fonctionnement de la stratification multi-label.

---

## ## Correspondance Vidéo → Frame → Fold

L’organisation est basée sur une propagation stricte des labels et du fold de la vidéo vers ses frames :

```text
Vidéo : d16427.mp4
├── Labels : [3, 12]
├── Fold : 2
└── Frames :
d16427_f00.jpg → fold 2
d16427_f01.jpg → fold 2
d16427_f02.jpg → fold 2
d16427_f03.jpg → fold 2
d16427_f04.jpg → fold 2
````

Cette règle élimine tout risque de contamination entre ensembles d’entraînement et de validation.

## Gestion des frames sans labels ou sans fold

Lors de la génération du fichier `frames_5_forstep1and2_folds.csv`, certaines frames apparaissent sans labels et sans numéro de fold. Ce comportement est normal et lié à la structure du dataset.

Le fichier d’annotations `track1-qv_pipe_train.json` ne contient que les vidéos du jeu d’entraînement annoté. Le dossier `data/raw_videos/` peut en revanche contenir davantage de vidéos, notamment :

- des vidéos du jeu de test (non annoté),
- des vidéos non incluses dans le sous-ensemble annoté,
- des fichiers résiduels ou issus d’extractions complètes.

Lorsque les frames sont extraites à partir de l’ensemble des fichiers MP4 présents, certaines proviennent donc de vidéos qui ne figurent pas dans le JSON. Lors du merge vidéo ↔ frames, ces frames ne trouvent pas de correspondance et reçoivent des champs vides (`labels_str` et `fold`).

Ces frames non annotées :

- ne sont pas utilisées pour l’entraînement,
- n’interviennent pas dans la construction des folds,
- n’introduisent aucune fuite d’information,
- peuvent être conservées ou supprimées sans impact sur la validité du pipeline.

Seules les vidéos annotées sont prises en compte dans la stratification et dans les DataLoaders d’entraînement/validation.

## ## Rôle de ces fichiers dans le pipeline complet

Les splits 5-fold obtenus sont utilisés dans les étapes suivantes :

1. **Entraînement frame-wise (baseline)**  
   Utilisation des frames extraites et des folds pour réaliser un premier modèle simple.

2. **Construction des super-images (3×3)**  
   Association des folds avec les images composées créées à partir des frames.

3. **Entraînement 5-fold complet**  
   Mise en œuvre des stratégies utilisées dans les modèles performants :  
   - Optimiseur AdamW  
   - OneCycleLR  
   - Pertes ASL ou CB-Focal  
   - Évaluation multi-label via mAP

4. **Ensemble final (fusion des modèles)**  
   Combinaison pondérée des 5 modèles issus des 5 folds.

---

## ## Résumé

La création des splits 5-fold stratifiés multi-label permet :

- une répartition équilibrée des classes,  
- une gestion correcte des labels multiples,  
- une validation fiable conforme aux standards actuels,  
- une préparation optimale pour les phases d’entraînement suivantes.

Cette étape constitue un pilier fondamental du pipeline d’apprentissage du projet.

---
