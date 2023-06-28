# Glossaire
Glossaire des termes liés au LLM.

## Modèle de language géant (Large Language Model, LLM)
Un réseau de neuronnes artificiel

## Réseau de neuronnes artificiel (Artificial Neural Network, ANN)
Les réseaux de neuronnes artificiels tentent de répliquer la structure du cerveau (des neuronnes, reliés entre eux par des connexions, les synapses). Un ANN est une succession de couches de neuronnes ; chaque connexion compte pour un paramètre, que l'on appelle _poids_. Plus le poids d'une connexion est importante, plus la connexion est forte, et plus le neuronne en sortie sera stimulé lors de la stimulation du neuronne d'entrée.

La structure la plus basique d'un ANN est la structure _[Feed Forward]()_, ou tous les neuronnes composant chaque couche sont reliés à tous les neuronnes de la couche suivante. Il existe évidemment des structures plus complexes de ANN, utilisés en _Computer Vision_ comme en _Natural Language Processing_.

## Modèle fondation (Fundation Model)
Un LLM ayant été suffisamment entrainé pour comprendre les propriétés fondamentales du language naturel : structure des phrases, raisonnement de bases, etc. Cependant, un modèle fondation n'est **pas encore prêt à être utilisé** car il n'a pas encore été entrainé à réaliser des taches concrètes, comme le résumé de textes, la traduction, l'analyse de sentiments...

GPT1 a montré qu'il est plus efficace de d'abord pré-entrainer des LLM sur des datasets de grande taille, avant de le spécialiser dans des taches concrètes à travers une étape de [Fine-tuning]() (plus d'informations disponibles dans cette section).

**C'est autour de ces modèles fondations que se situent aujourd'hui les principaux enjeux** autour des LLM open-source. Constituer un jeu d'entrainement puis pré-entrainer un LLM à plusieurs centaines de millions de paramètres requiert une puissance de calcul très importante. De ce fait, les grands projets tels que [Bloom]() au niveau européen, ou plus récemment [Falcon]()

De fait, les grands projets open-source tels que [Bloom](), ou plus récemment [Falcon]()

![](69d98221-ab76-4061-8a11-a2669c2ab5d2_2048x1632.jpg)