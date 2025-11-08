Nettoyez vos colonnes mot1, mot2 (comme vous faites déjà).
Encodez mot1 et mot2 individuellement : par exemple via un embedding de mot (Word2Vec / FastText) ou un modèle de langue (ex : BERT) pour récupérer un vecteur.
Combinez ces deux vecteurs (ex : concaténation, différence, produit, etc.).
traitez la relation comme label.
Si vous classifiez la relation : entraînez un réseau (MLP, ou un modèle Transformer léger) pour prédire la relation à partir des vecteurs combinés.