# batcave
This is the batcave, full of experiments and weird manipulations.

# Requirements 
Veuillez installer tous les modules nécessaires en faisant :
```bash
pip3 install -r requirements
```

Puis il faut ensuite installer le package "`punkt_tab`" utilisé par `nltk`. Pour ça, ouvrez une instance Python3 sur un terminal en faisant `ipython`, comme dans l'exemple suivant :
```bash
franzele@franzele-HP-ProBook:~$ ipython
Python 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import nltk

In [2]: nltk.download("punkt_tab")
[nltk_data] Downloading package punkt_tab to
[nltk_data]     /home/franzele/nltk_data...
[nltk_data]   Package punkt_tab is already up-to-date!
Out[2]: True
```

# Utilisation

Pour utiliser l'application, il suffit d'ouvrir un terminal à la racine du projet, et de faire la commande suivante :
```bash
cd scripts
uvicorn main:app --reload
```

Puis, vous devez aller sur le lien suivant [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Puis, pour entraîner le modèle, cliquez sur "train"->"Try it out" et remplacez le `"/path/to/train_dataset.csv"` du 
```
{
  "file_path": "/path/to/train_dataset.csv"
}
```

Par le chemin de votre fichier d'entraînement.

Pour tester le modèle, il faut l'avoir entraîné, puis cliquez sur "test"->"Try it out" et comme pour l'entraînement, remplacez le `"/path/to/train_dataset.csv"` par le chemin de votre fichier de test.

Pour tester un seul exemple, vous pouvez utiliser "predict", et vous pouvez remplacer les champs par vos propres données.