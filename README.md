<p align="center">
  <img src="assets/openllm.png" alt="OpenLLM"/>
  <br>
  OpenLLM Glossary
</p>

## Modèle de language géant (Large Language Model, LLM)
Un réseau de neuronnes artificiel destiné à modéliser le language naturel, et dôté d'un très grand nombre de paramètres numériques (540Md pour Google PaLM, 170Md pour GPT3, 40Md pour Falcon).

## Réseau de neuronnes artificiel (Artificial Neural Network, ANN)
Les réseaux de neuronnes artificiels tentent de répliquer la structure du cerveau (des neuronnes, reliés entre eux par des connexions, les synapses). Un ANN est une succession de couches de neuronnes ; chaque connexion comptant pour un paramètre, que l'on appelle _poids_. Plus le poids d'une connexion est grand, plus la connexion est forte, et plus le neuronne de sortie répondra positivement à une forte stimulation du neuronne d'entrée.

> La structure la plus basique d'un ANN est la structure _[Feed Forward]()_, ou tous les neuronnes composant chaque couche sont reliés à tous les neuronnes de la couche suivante. Il existe évidemment des structures plus complexes de ANN, utilisés en _Computer Vision_ comme en _Natural Language Processing_.

<p align="center">
  <img src="assets/feed_forward.png" alt="Feed Forward Neural Network"/>
  <br>
  Un ANN simple (<a href="https://deepai.org/machine-learning-glossary-and-terms/feed-forward-neural-network">source</a>)
</p>

> Lorsqu'on parle "d'entrainer un modèle", on fait référence au fait de rechercher itérativement la valeur optimale de ses paramètres. Le procédé mathématique associé est la _déscente de gradient stochastique_.

## Modèle fondation (Foundation Model)
Un LLM ayant été suffisamment entrainé pour comprendre les propriétés fondamentales du language naturel : structure des phrases, raisonnement de bases, etc. Cependant, un modèle fondation n'est **pas encore prêt à être utilisé** car il n'a pas encore été entrainé à réaliser des taches concrètes, comme le résumé de textes, la traduction, l'analyse de sentiments...

GPT1 a montré qu'il est plus efficace de d'abord pré-entrainer des LLM sur des datasets de grande taille, avant de le spécialiser dans des taches concrètes à travers une étape de [Fine-tuning]() (plus d'informations disponibles dans cette section).

**Ce sont les modèles fondation qui concentrent les enjeux des LLM open-source aujourd'hui.** Constituer un jeu d'entrainement puis pré-entrainer un LLM à plusieurs centaines de millions de paramètres requiert une puissance de calcul très importante. De ce fait, les grands projets comme [Bloom]() ou plus récemment [Falcon]() visent à entrainer des modèles fondation open-source, prêts au [Fine-tuning](#), de sorte à ce que les chercheurs ou universitaires n'ayant pas accès à de tels moyens puissent tout de même faire travailler sur ces modèles massivement pré-entrainés, afin de faire avancer le domaine des LLM.

## Fine-Tuning
Aujourd’hui, les LLMs sont d’abord pré-entrainés sur de gigantesques datasets d’entrainement grâce à l'apprentissage semi-supervisé (sans l'invervention systématique de l'homme). Ces datasets d'entrainement sont collectés puis filtrés automatiquement grâce à Internet : blogposts, articles scientifiques, réseaux sociaux...

Seulement, ce pré-entrainement à lui seul n’est pas assez efficace pour apprendre au modèle à effectuer une tache spécifique concrète (appelée _downstream task_), comme la traduction, le résumé de texte... Souvent, ce pre-training n’apprend au modèle que des propriétés générales du langage : signification du vocabulaire, structure des phrases, raisonnements de base...

Le fine-tuning consiste à entrainer dans un second temps un LLM pré-entrainé, dans le but de lui apprendre une de ces taches particulières. Cette étape est alors bien moins coûteuse que le pré-entrainement, car le modèle a déjà assimilé les propriétés générales du langage.

> Souvent, cette étape de l'entrainement est réalisée avec un dataset plus réduit en taille, car les données nécessaires nécessitent l'intervention humaine : écriture de résumés, évaluation de textes générés,  labélisation des données...

<p align="center">
  <img src="assets/training_model.png" style="width: 80%;" alt="Entrainement standard d'un LLM"/>
  <br>
  Entrainement standard d'un LLM (<a href="https://fatyanosa.medium.com/fine-tuning-pre-trained-transformer-based-language-model-c542af0e7fc1">source</a>)
</p>

> Cette pratique n'est pas du tout pas spécifique au NLP : en _Computer Vision_ par exemple, la technique des _LoRA_ permet d'apprendre aux fameux modèles de type _Stable Diffusion_ des connaissances picturales sans avoir à ré-entrainer tout le modèle (seulement une infime partie de connexions judicieusement positionnées).

## Instruction-Tuning
Il s'agit d'un type spécifique de fine-tuning. On apprend au modèle à bien réagir face à des instructions du type "Peux-tu me résumer le texte suivant: [...]" ou encore "Traduit en anglais la phrase suivante: [...]". Il s'agit d'instructions que l'utilisateur pourrait vouloir fournir à un ChatBot tel que ChatGPT pour intéragir avec un modèle de manière conversationnelle.

## Apprentissage par renforcement avec retour humain (Reinforcement learning from Human Feedback, RLHF)
Le reinforcement learning fait référence à une technique d'apprentissage où le LLM évolue dans un environnement fictif : cet environnement fournit à chaque instant un état au LLM, qui effectue des décisions en fonction desquelles il se voit récompenser. Il s'agit de la troisième technique majeure d'apprentissage, en plus de _l'apprentissage supervisé_ et de _l'apprentissage non supervisé_.

Le RLHF consiste concrètement en une boucle fermée où les éxaminateurs stimulent le modèles puis en corrige les prédictions afin de l'améliorer continuellement. En pratique pour les LLM, il est principalement utilisé en conjonction avec un entrainement type [Fine-tuning](), par exemple pour bannir certaines réponses jugées dangeureuses ou vulgaires.

## Underfitting, Overfitting

L'overfitting est un problème rencontré lorsque le modèle trop adapté à un dataset particulier. Dès lors, il sous-performera lorsque utilisé en dehors de sa "zone de confort" par rapport à un même modèle, moins entrainé. Pour résoudre l'overfitting, il faut moins entrainer le modèle ou augmenter la qualité (taille, diversité) du dataset pour le rendre plus complexe.

L'underfitting est simplement l'inverse, cela fait référence à un [ANN]() qui sous-performe car il n'a pas été assez entrainé sur un dataset particulier : il n'arrivera alors pas à en capter toutes les subtilités et renverra une "réponse approchée" de faible qualité. Pour résoudre l'underfitting, il faut plus entrainer le modèle.

<p align="center">
  <img src="assets/fitting.PNG" style="width: 80%;" alt="Overfitting et Underfitting"/>
  <br>
  Overfitting et Overfitting (<a href="https://fr.mathworks.com/discovery/overfitting.html">source</a>)
</p>

## Transformeur (Transformer)
Structure particulière de [modèle de language géant](). [Depuis 2017](https://arxiv.org/abs/1706.03762), il s'agit de la famille de LLMs avec laquelle on obtient les meilleures performances dans le traitement du langage naturel dans toutes les tâches communes : traduction, ChatBot, résumé de texte, . Il a remplacé les _réseaux de neurones récurrents_ (RNN) dont l'un des points faibles est l'établissement de relations entre des mots distants.

Son atout est son système d'[attention]() qui lui permet d'exploiter les relations entre les mots (notamment les références telles que "il" ou "ce dernier"), pour inférer la signification d'une phrase.

La structure initiale transformeur est composée une pile d'[encodeurs]() connectée à une pile de [décodeurs](), même si cette dernière a été largement adaptée depuis. Aujourd'hui, les LLM les plus performants suivent une architecture type de [Decoder-Only]() même si les structures de type [Encoder-Only]() sont encore utilisées pour la classification ou la mesure de performances de modèles, entre autres.

<p align="center">
  <img src="assets/transformer_model.png" style="width: 50%;" alt="Structure initiale du transformeur"/>
  <br>
  Structure initiale du transformeur (<a href="https://arxiv.org/abs/1706.03762">source</a>)
</p>

## Modèles Encoder-Only (Encoder-Only Models)


## Modèles Encoder-Decoder (Encoder-Decoder Models)
Il s'agit de la structure qui avait été présentée dans le [papier initial du transformeur](https://arxiv.org/abs/1706.03762). Ce modèle est composé d'un [encodeur](), dont la fonction est d'encoder la phrase i.e en trouver une représentation vectorielle qui tient compte de sa signification, et d'un [décodeur](), dont la fonction est de générer un texte, étant donné le contexte fourni par l'encodeur.

Les modèles Encoder-Decoder ne sont majoritairement plus développés, étant donné que de meilleures performances sont obtenus plus simplement par les modèles de type Decoder-Only. On peut tout de même citer [BART](https://arxiv.org/abs/1910.13461) qui est resté pendant quelques mois l'état de l'art en 2019 en texte de tâches génératives, ou encore [T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) (_Text-to-Text Transfer Transformer_) dont la spécifité est de traiter toutes les _drownstream tasks_ en langage naturel brut ("traduit en français: [...]" ou "résume ce texte: [...]").