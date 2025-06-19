# Fissures – Module Route : Suivi analytique d’une fissure murale

> Partie autonome du projet **Fissures**.  
> Produit un tableau de bord Dash pour explorer les mesures du comparateur numérique placé sur le mur « Route ».

---

## Sommaire
1. [Aperçu fonctionnel](#aperçu-fonctionnel)  
2. [Architecture du dépôt](#architecture-du-dépôt)  
3. [Installation & environnements](#installation--environnements)  
4. [Configuration](#configuration)  
5. [Lancement rapide](#lancement-rapide)  
6. [Branching model & workflow Git](#branching-model--workflow-git)  
7. [Convention de commits](#convention-de-commits)  
8. [Tests & intégration continue](#tests--intégration-continue)  
9. [Feuille de route](#feuille-de-route)  
10. [Licence](#licence)

---

## Aperçu fonctionnel
* **Ingestion** : téléchargement automatique du CSV depuis le Raspberry Pi via SSH/SFTP (backup journalier).  
* **Nettoyage** :  
  * filtre d’intervalle ≥ 10 s pour supprimer les points redondants d’un même palier,  
  * suppression manuelle de deux valeurs corrompues (timestamps listés dans `main_route.py`).  
* **Statistiques** :  
  * moyennes, médianes, min, max et intervalles de confiance bootstrap (95 %) pour chaque jour,  
  * détection robuste des heures & valeurs extrêmes (LOWESS + raffinage),  
  * ajustement de distributions (von Mises, logistique, Weibull, etc.).  
* **Visualisation** : Dash multi‑onglets — courbes historiques, histogrammes/ KDE, QQ‑plots, jours moyen & médian.  
* **Règles métier** :  
  * le **dernier jour** (jour courant) est toujours exclu car incomplet,  
  * le **premier jour** est inclus **sauf** si la date de début d’analyse vaut `2025‑03‑15` (date min du flux).

---

## Architecture du dépôt
```text
route/
├── data/
│   └── measurements.csv          # Copie locale du CSV (mise à jour auto)
├── config.yaml                   # Paramètres (chemins, SSH, seuils…)
├── main_route.py                 # Entrée Dash + pipeline
├── data_loader.py
├── aggregation.py
├── time_calculator.py
├── stats_calculator.py
├── figures.py
├── tests/                        # Pytest unitaires
│   └── ...
├── requirements-route.txt        # Dépendances Python pour ce module
└── README.md
```

## Installation & environnements

``` bash

# 1) Cloner le dépôt général puis se placer dans le sous-dossier route
git clone https://github.com/johann-DP/Fissure/blob/master/data/Fissures

# 2) Créer un venv dédié
python -m venv .venv-route
source .venv-route/bin/activate     # Windows : .\.venv-route\Scripts\activate

# 3) Installer les dépendances
pip install -r requirements-route.txt
# ou (optionnel, futur) : uv pip install -r requirements-route.txt
```

```
⚠️ Les autres parties du projet utilisent un venv différent (versions pandas & Python non compatibles). Ne pas mélanger.
```

## Configuration
1. Renommer config-example.yaml en config.yaml.
2. Remplir les sections :
```yaml
ssh:
  host: raspberrypi
  port: 22
  user: johan
  password: "••••••"
paths:
  local_csv: data/measurements.csv
  remote_csv: /home/johan/measurements.csv
analysis:
  start_day_default: 2025-03-31
  min_interval_seconds: 10
  window_half_width: 3.0
logging:
  level: INFO
  file: app.log
```

## Lancement rapide
```bash
# Avec la date par défaut (config.yaml)
python ./data/Fissures/'Fissure route'/data_route/main_route.py

# Ou en choisissant la date de départ
python ./data/Fissures/'Fissure route'/data_route/main_route.py --start-date 2025-03-31
```

Le serveur Dash démarre sur http://localhost:8051.
L’analyse met à jour la sauvegarde locale, calcule les statistiques et affiche les graphiques sans modifier leur organisation d’origine.

---
## Branching model & workflow Git

- master : production (Dash opérationnel).
- develop : intégration continue.
- feature/<nom> : one small change (voir feuille de route).
- release/<vx.y.z> : stabilisation avant merge sur master.
- hotfix/<descr> : réparation urgente en production.

Exemple de cycle :
```
# créer une nouvelle fonctionnalité
git checkout develop
git checkout -b feature/config-yaml

# travailler, committer…
git add config.yaml config.py
git commit -m "feat[build]: add yaml config loader"

# push & MR vers develop
git push origin feature/config-yaml
# GitLab / GitHub : ouvrir la merge‑request

# après revue :
git checkout develop
git merge --no-ff feature/config-yaml
git push origin develop
```
---

## Convention de commits

```php-template
<type>[<portée>]: <sujet>
<description motivant le changement>

Footer : (optionnel) version, close #issue, etc.
```

### Types admis
```
build, ci, docs, feat, fix, perf, refactor, style, test
```

### Exemples
- feat[Pre-processing]: add yaml config loader
- perf[DataViz]: vectorize calculate_daily_stats
- fix[Model]: correct bootstrap confidence interval

---

## Tests & intégration continue
```bash
# Lancer la suite locale
pytest -q

# CI (GitHub Actions)
.
├── lint   : flake8 + black --check
├── tests  : pytest
└── build  : démarrage Dash headless (timeout 30 s)
```
Tout merge vers develop ou master doit passer la CI.

---

## Feuille de route (extrait)

| Sprint | Feature                         | Branche                                       | Tag prévu |
| ------ | ------------------------------- | --------------------------------------------- | --------- |
| 1      | YAML config + logger de base    | `feature/config-yaml`, `feature/logging-base` | `v0.1.0`  |
| 2      | CLI `--start-date`              | `feature/cli-start-date`                      | `v0.2.0`  |
| 3      | Bootstrap IC (remplace Student) | `feature/bootstrap-ci`                        | `v0.3.0`  |
| 4      | Vectorisation pandas & tests    | `feature/vector-stats`                        | `v0.4.0`  |

---

## Licence
© 2025 Company / Author. Tous droits réservés. Licence interne à définir.


---

### Remarque finale
*Ces documents intègrent toutes les clarifications (premier ≠ dernier jour, présence des doublons graphiques, gitflow basé sur `master`, coexistence de plusieurs venv, etc.).  
Ils peuvent être committés immédiatement (branche `feature/docs-readme`) sans impacter le Dash.*
