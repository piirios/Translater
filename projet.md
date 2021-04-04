#### Description
[Translater](https://github.com/piirios/Translater) est un mini-projet ayant pour but de créer un traducteur d'une langue à une autre. Il n'a pas pour philosophie de devenir le meuilleur traducteur actuel mais uniquement d'être un projet cool à créer. Le projet est composée d'un modèle basée sur les [Transformers](https://arxiv.org/abs/1706.03762) et j'utilise directement l'implémentation faite par [pytorch](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer).Le modèle est entrainée grâce au dataset fournis par le [parlement européens](https://www.mllp.upv.es/europarl-st/).

#### Installation
##### Telecharger le projet
soit vous passez par ce [lien](https://github.com/piirios/Translater) pour télécharger le projet
soit vous effectuer dans une console
```shell
git clone https://github.com/piirios/LMcommand.git
```
##### Installer les dépendances
dans le dossier du projet effectuer
```shell
pip install -r requirement.txt
```
##### Dataset
telecharger le dataset sur ce [lien](https://www.mllp.upv.es/europarl-st/v1.1.tar.gz)
ensuite deziper-le et dans le répertoire du projet effectuer
```shell
python cli.py set_dataset_folder FOLDER
```
où FOLDER est le chemin du dossier racine du dataset cad. le chemin du  dossier parent de v1.1 
##### Dossier des poids
dans le répertoire du projet effectuer
```shell
python cli.py set_dataset_folder FOLDER
```
où folder est le dossier où vous voulez sauvegarder les poids du modèles
### Utilisation
#### CLI
##### Entreinement
Pour entreinez un traducteur effectuer
```shell
python cli.py train -fl FROM_LANGUAGE -tl TO_LANGUAGE
```
où FROM_LANGUAGE est la langue source et TO_LANGUAGE est la langue ciblée
vous pouvez aussi spécifier différent paramètres du modèle:
 * "--num-epochs" qui est le nombre d'epochs du l'entreinement   (defaut: 10000)
 * "--batch-size" qui est la taille de chaque batch (default: 32)
 * "--learning-rate" qui est le coéficient d'apprentissage (defaut: 3e-4)
 * "--max-len" qui est la taille maximale des phrases traduites (defaut: 200) 
/!\ Je déconseille de réduire "--max-len" car vous pouvez avoir des soucis lors de l'entreinement

le reste des paramètres sont propres au Translaters, pour de plus compréhenssion je vous renvoi à la document de pytorch sur ce [sujet](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer). Voici la liste de ces paramètres:
 * "--embedding-size" (defaut: 512)
 * "--num-heads" (defaut: 8)
 * "--num-encoder-layers" (defaut: 3)
 * "--num-decoder-layers" (defaut: 3)
 * "--dropout" (defaut: 0.1)
 * "--forward-expansion" (defaut: 200)

##### Traduction
Pour traduire une phrase effectuer
```shell
python cli.py translate -fl FROM_LANGUAGE -tl TO_LANGUAGE -sq SENTENCE
```
où FROM_LANGUAGE est la langue source, TO_LANGUAGE est la langue ciblée et SENTENCE est la phrase que vous souhaiter traduire.

#### Importation
Vous pouvez imcorporer ce projet dans vos programmes python directement via l'importation
```python
from .main import Translater
translater = Translater("FL", "TL")
translated = translater("SQ")
```
où FL est la langue source, TL est la langue ciblée et SQ est la phrase que vous souhaiter traduire.

