# Project: Enabling More Effective Learning of Languages via Personalized Extensive Reading
Group Members: Zihao Wang, Jinjin Yu
![Spiderman](/Image/Spiderman.png)

## Repository Index
- [Language leanrning_Spiderman.ipynb](/Language%20learning_Spiderman.ipynb): The main python notebook for this assignment
- [Image](/Image): The folder of essential images in the notebook
- [best_model.pkl](/best_model.pkl): The model with the best performance for classification in this assignment
- [french.gexf](/french.gexf): The network file which can be imported into Gephi to adjust the layout, nodes, edges, etc.
- [credentials.json](/credentials.json): The personal secret key for Youtube API
- [streamlit.py](/streamlit.py): The python file to display the interface (run `streamlit run streamlit.py` in command line or terminal)
- [training_data.csv](/training_data.csv): The original data for training (provided by Kaggle)
- [unlabelled_test_data.csv](/unlabelled_test_data.csv): The unlablled data for final prediction and uploading onto Kaggle (provided by Kaggle)
- [max-dataset.csv](/max-dataset.csv): The additional data from outer sources
- [outcome.csv](/outcome.csv): The outcome of the text difficulty classification using our best model
- [sample_submission.csv](/sample_submission.csv): The sample of submission to Kaggle (provided by Kaggle)
- [token.pickle](/token.pickle): The auto-generated token through the process of accessing API
- [catboost_info](/catboost_info): The auto-generated catboost information when training the Catboost model


## Video
 **[Access our video here through Google Drive](https://drive.google.com/file/d/1ahaZZMjGMmbnw7lGuf23vuvZMUwMBchs/view?usp=share_link)**

## 1 Project Overview
### 1.1 Problem Description

Extensive reading is essential for language learning, especially audio and video materials that effectively improve listening and speaking skills. It offers language learners the opportunity to be exposed to a wide range of vocabulary, to understand and become familiar with different grammatical structures and sentence patterns, and to improve reading comprehension and reasoning skills. However, many French language resources currently on the market, take YouTube videos as an example, do not have a hierarchy of difficulty. Also, there is a lack of a system, which can give appropriate recommendations according to the learners' level and interests. The YouTube website has a large corpus of resources, and subtitles provide us with a medium for analyzing videos. To explore solutions to the problems above with YouTuBe, we conducted this project.

### 1.2 Sustainability

This project is closely related to many topics in the SDGs:

- [Quality Education (Goal 4)](https://sdgs.un.org/goals/goal4): By providing personalized reading materials that match learners' interests, the project is able to increase learner motivation and engagement. In addition, the project focuses on providing suitable learning materials of appropriate difficulty according to the learners' ability level, which can effectively serve as supplementary materials for the systematic learning and help learners to improve their learning efficiency. Meanwhile, this project serves as a practical application of personalized recommendation in language study, which has the potential to be expanded to fields such as arts, science, and programming learning.

- [Industry, Innovation, and Infratructure (Goal 9)](https://sdgs.un.org/goals/goal9): The replacement of traditional paper-based teaching methods with internet and electronic resources has improved the sustainability of resources. Educational resources can also be disseminated to a broader range of areas, allowing more people to benefit from language learning. On the other hand, the digital transformation of the education industry has ignited creators' passion and promoted educational innovation and diversity.

- [Sustainable Cities and Communities (Goal 11)](https://sdgs.un.org/goals/goal11): The rich educational resources on YouTube are open to everyone, irrespective of gender, race, or social status, ensuring equitable access to learning materials for all individuals. Language learning and training can elevate their educational levels and employment opportunities, thereby fostering community development and prosperity. Simultaneously, through cross-cultural reading experiences, learners can gain a better understanding of and appreciation for cultural diversity, facilitating cross-cultural communication and understanding.

### 1.3 Overall Objective

To solve the problems mentioned above, this project consists of 3 main parts:
- The first part mainly focuses on the text classification problem.
- The second part mainly focuses on the topic clustering problem.
- The third part mainly focuses on the interactive recommendation problem.

Finally, we reflect on and discuss the entire project's process and results and extend them.

## 2 Data Preparation
Training data: 4800 rows of original data + 9174 rows of expanded data.

Final test data: 1200 rows of data. 

Due to the small sample size of the provided training set (4800 rows), we supplemented it with 9174 rows of data from external sources to improve the accuracy of the model. At the same time, since this text classification task only requires text feature extraction and does not include other features beyond text (e.g., context, text narrator information, etc.), we only need to perform de-duplication. After that, the training data consists of 10,871 rows, which will be used for the training of subsequent classification models and topic clustering models. 

We also conducted EDA of the dataset, mainly with regard to the average number and size of each class.

| Difficulty | Mean Length | Count |
|---------|---------|---------|
| A1 | 42.898670 | 1579 |
| A2 | 64.973542 | 2041 |
| B1 | 88.757473 | 1773 |
| B2 | 116.749716 | 1758 |
| C1 | 147.102167 | 1938 |
| C2 | 201.145342 | 1782 |

We found that each label has a relatively evenly distributed amount of text, ranging between 1500 and 2100. However, text length increases with increasing difficulty level. Longer texts may contain more information and semantic details, which may require more sophisticated feature extraction methods and models to capture the important features of the text, increasing the difficulty of prediction. It's noteworthy that as we haven't done any processing to the texts (e.g. tokenization), the texts are likely to be longer than the tokenized ones. At the same time, the box plot reflects that texts at the C2 level show greater variability and diversity in terms of length. However, they are not necessarily incorrect or invalid data, but may be due to the fact that higher-ranked texts cover more topics and domains and contain more complex sentence structures and longer descriptions.

![EDA-Length](/Image/EDA-Length.png)
![EDA-LengthDistribution](/Image/EDA-LengthDistribution.png)

## 3 Text Difficulty Classification
### 3.1 Objective
This part aims to predict the difficulty level to which a text (in French) belongs based on its features and properties. Specifically, it is to achieve the following objectives:
- Extract the features of the text: Convert the text into a numerical feature vector to effectively capture the semantics of it.
- Select the appropriate classifier: The type of classifier has a significant impact on the accuracy of the prediction results, so the best performing trained model needs to be used.
- Predicting the difficulty level of the text: The extracted text feature vector is used as input to predict the difficulty level for classification using the rained model. Observe the generalization ability of our model.

### 3.2 Literature Overview
Text difficulty classification is an important task in natural language processing and is widely used in the fields of education, linguistic research and computer-assisted learning. Researchers in this field mainly focus on two aspects: linguistic feature-based methods and machine learning-based methods.

The linguistic feature-based approach determines the difficulty level of a text by analyzing its linguistic attributes. Researchers typically use a range of text metrics to measure these attributes, such as lexical complexity, sentence structure, chapter organization, and contextual relevance. They can then build models to predict the difficulty level. Menglin Xia et al. ([2019](https://arxiv.org/abs/1906.07580)) applied a generalization method to adapt models trained on larger native corpora to estimate text readability for learners, which achieves an accuracy of 0.797. The research of Josiane Mothe et al. ([2005](https://shs.hal.science/halshs-00287692/)) demonstrated the importance of syntactic complexity and word polysemy for linguistic features. The results also opened the way for a more enlightened use of linguistic processing in IR systems. Scott A. et al （[2007](https://escholarship.org/content/qt39r3d755/qt39r3d755.pdf)）used the Coh-Metrix computational tool and different layers of language, text difficulty of discourse, and conceptual analysis as means of measuring text readability in English. Other similar traditional methods include the New Dale–Chall Readability Formula (Stocker et al. [1971](https://www.proquest.com/docview/1994302962?pq-origsite=gscholar&fromopenview=true&imgSeq=1)), the Lexile framework (Smith et al. [1989](https://eric.ed.gov/?id=ED307577)), the Advantage-TASA Open Standard for Readability (ATOS) formula (Eunkyung Hwang et al. [2019](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE10715205)), etc.

Machine learning-based approaches are the mainstream research direction nowadays. It uses machine learning algorithms to train models that can predict the difficulty level of a text based on its features. These methods usually use supervised learning methods and require the use of labeled datasets to train the models. Sarah et al. ([2005](https://aclanthology.org/P05-1065.pdf)) combined support vector machines with traditional reading level measures, statistical feature language models, and other language processing tools to improve the accuracy of the assessment. Renu et al. ([2020](https://link.springer.com/article/10.1007/s40593-020-00201-7)) found that incorporating additional linguistic features from NLP tools improved ML classification accuracy by over 10% compared to simple readability metrics. Arun et al. ([2019](https://link.springer.com/article/10.1007/s10669-019-09717-3)) analyzed different models' performance metric biases and found that active learning has the disadvantage of generating model performance prediction biases. They proposed that using a randomly selected independent validation dataset could reduce the bias in model performance prediction. Utomo et al. ([2019](https://ieeexplore.ieee.org/abstract/document/8884317)) used K-mean clustering and polynomial plain Bayesian methods based on Lexile Level for text difficulty classification and prediction, achieving good performance.

In recent years, with the development of deep learning, neural network-based approaches have made significant progress in text difficulty classification tasks. These methods utilize neural network models, such as Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RN), to learn the representation and features of text. Leveraging the powerful representational power of deep learning models, these methods have achieved high accuracy results in such tasks. Kamran et al. ([2017](https://ieeexplore.ieee.org/abstract/document/8260658?casa_token=lKWYEuuOrrIAAAAA:mQL2jEaJWD3VJGuFBcEmSEAiMAVqZ5I7RPKKvvK_CTCAp5nf4fKPc9EWIAXNvR4cVitbBOnZja4)) improved the performance of classifiers for processing large amounts of text with Hierarchical Deep Learning for Text classification (HDLTex). M. Ghiassi et al. ([2012](https://www.sciencedirect.com/science/article/pii/S0957417412004976?casa_token=wymV3rhU53cAAAAA:5zitKb3FTPBHIL3lDqUZ1IjN64vwTsLwdLSktpB0tNHGj1LKowSy_chp4cBUBiDWQAgxS80NY0M)) introduced dynamic architecture for artificial neural networks (DAN2) as an alternative for solving textual document classification problems. EM Dharma et al. ([2022](http://www.jatit.org/volumes/Vol100No2/5Vol100No2.pdf)) compared the performance of Word2Vec, GloVe, and FastText when using convolutional neural network algorithms for text classification and found more consistent accuracy.

### 3.3 Processiong
#### 3.3.1 Tokenization and Text Feature Extraction (Bert)
At the very beginning, we created `X = train['sentence']` and `y = train['difficulty']` for model training.

Bidirectional Encoder Representations from Transformers (Bert), is a pre-trained language model based on the Transformer architecture. It can be used for a variety of natural language processing (NLP) tasks by learning rich language representations through unsupervised training on a large-scale text corpus. Considering Bert's powerful pre-trained language representation, bi-directional coding capability, and migration learning capability, we selected it for text feature extraction at this stage.

To implement this method, we defined a function called `bert_feature(data, **kwargs)`. This function accepts a list of data as input and performs tokenization and encoding operations. It iterates through the list of input texts and proceeds the following steps for each text: 1) Transform the text into a sequence of model-acceptable input ID. 2) Pass the input ID tensor to the embedding layer of the Camembert model to transform it into embedded features. 3) Extract the embedded features using the forward method of the Camembert model. In this code, we extract only the first embedding vector of each input sequence. 4) Add the extracted features to the feature list. At the end of the loop, all features are concatenated from the list to form an array of feature data. Finally, the function returns the feature data array and empties the cache using `torch.cuda.empty_cache()`.  As a result, we successfully obtain 768 feature vectors `X = bert_feature(X)` of the input text. Partial results are shown below. 

![Bert_Features](/Image/Bert_Features.jpeg)

In order to find the most appropriate model for this task, we tried Camembert, Flaubert, and Multilingual methods respectively and compared the ultimate accuracies of them. The tabel below displays the ccuracy of the best performing classifier under each model.
|         | Camembert (Extra Trees) | Flaubert (Catboost) | Multilingual (XGBoost) |
|---------|---------|---------|---------|
| Accuracy | 0.624081 | 0.538603 | 0.544118 |
| Precision | 0.636840 | 0.547111 | 0.551744 |
| Recall | 0.623988 | 0.540955 | 0.541880 |
| F1 | 0.627376 | 0.543244 | 0.543796 |


The above results show that the Extra Trees classifier based on Camembert model has the highest accuracy and best performance. We speculate that this is due to the fact that Camembert is a pre-trained language model developed specifically for French, which has a deeper understanding of the grammar, syntax, and semantics and can better capture the features of French text. Meanwhile, the model uses a vocabulary list dedicated to French, which contains French-specific words and tokens to reduce ambiguity. Flaubert, on the other hand, focuses more on different linguistic representational capabilities and contextual understanding; Multilingual is better suited to handle texts containing multiple languages.

#### 3.3.2 Select Classification Model
We trained the extracted features previously using different classifier models (SVC, logistic regression, random forest, etc.) and evaluated their performance on the test set (also part of the original training set).

Firstly, we divided the dataset into training set (x_train and y_train) and test set (x_test and y_test) proportionally `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)`. The ratio is 9:1. We initially trained at a ratio of 8:2 but found that the results were not as good as 9:1. This may be due to the high model complexity and limited amount of data, so using a larger ratio of training samples can provide more information and help improve the generalization ability of the model. Then, we used LabelEncoder to encode the target variables and converted them into numerical form. The numbers 0-5 represent the six difficulty levels: A1, A2, B1, B2, C1, and C2, respectively.

Next, we defined a function named `evaluation(model, X_test, y_test)` for evaluating the performance of the model. This function calculates accuracy, precisio, recall, and F1 values as evaluation metrics. These metrics measure the overall prediction accuracy, positive prediction accuracy, positive prediction coverage, and balance of the model, respectively.

We tried `SVC()`, `Logistic Regression()`, `RandomForestClassifier()`, `ExtraTreesClassifier()`, `KNeighborsClassifier()`, `LGBMClassifier()`, `XGBClassifier()`, and `CatBoostClassifier()` eight classifiers in turn for training. The accuracies of ExtraTrees, LightGBM, RandomForest, and XGBoost classifiers are significantly higher than those of others. For these four classifiers, we further selected the optimal parameters using GridSearch. Finally, the 'evaluate' function is called to display the performance of these models. The comparison results are as follows:

|  | SVC | KNN | Loxgistic Regression | Random Forest | Random Forest (best para) | Extra Trees | Extra Trees (best para) | LightGBM | LightGBM (best para) | XGBoost | XGBoost (best para) | Catboost |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Accuracy | 0.506434 |	0.536765 | 0.542279| 0.579044 | 0.611213 | 0.608456 | 0.624081 | 0.606618 | 0.617647 | 0.613051 | 0.609375 | 0.618566 |
| Precision | 0.514941 | 0.553980 | 0.543911 | 0.589596 | 0.626067 | 0.621664 | 0.636840 | 0.618743 | 0.628955 | 0.623097 | 0.613764 | 0.626426 |
| Recall| 0.504060| 0.538852 | 0.544995 | 0.578662 | 0.610947 | 0.609443 | 0.623988 | 0.607830 | 0.618681 | 0.614737 | 0.611237 | 0.619721 |
| F1 | 0.505178 | 0.537776 | 0.543103 |	0.581227 | 0.614183 | 0.613173 | 0.627376 |	0.611910 | 0.622191 | 0.617457 | 0.612133 | 0.622355 |

It can be observed that the Extra Trees classifier with optimal parameters has the best performance `etc_model_best = ExtraTreesClassifier(min_samples_leaf=1, min_samples_split=2, n_estimators=300, random_state=42)`. It has the highest precision and accuracy, as well as F1 score. However, we also sacrificed recall rate to some extent.

At last, we displayed the confusion matrix of the prediction results of this classifier model (Extra Trees). 

![Confusion Matrix](/Image/Confusion%20Metrics.png)

The heat map helps us to understand more intuitively how well the model classifies each category and the possible misclassification. We observe that the model classifies most sentences correctly. To judge the prediction capability of the model on different situations, we went deep into each category to explore the capabilities of the model. Through the table, we can see that this model has the best prediction recall for the text of C2 level and the worst for B1.

|         | Precision | Recall | F1-score | Support |
|---------|---------|---------|---------|---------|
| 0 | 0.75 | 0.71 | 0.73 | 172 |
| 1 | 0.52 | 0.66 | 0.58 | 211 |
| 2 | 0.58 | 0.48 | 0.52 | 173 |
| 3 | 0.66 | 0.56 | 0.60 | 179 |
| 4 | 0.56 | 0.60 | 0.58 | 186 |
| 5 | 0.75 | 0.73 | 0.74 | 167 |
| accuracy | | | 0.62 | 1088 |
| macro avg | 0.64 | 0.62 | 0.63 | 1088 |
| weighted avg | 0.63 | 0.62 | 0.62 | 1088 |

#### 3.3.3 Prediction on the Test Set
Next comes the most exciting part: verifying the accuracy of our model on a real test set! Again, we first extract the features of the sentences in the test dataset using the pre-trained Camembert BERT model `test_features = bert_feature(test['sentence'])`. These features are then classified and predicted using the trained ExtraTrees classifier model, and the prediction results are converted to the corresponding difficulty level. The above results are saved to a CSV file `Name = 'outcome.csv'`. This file contains the ID of the test datasets and the corresponding difficulty levels.

In this project, we uploaded a total of 21 different classifier models to Kaggle. Through continuous tuning，the accuracy of the prediction results was improved from the initial 0.418 to 0.7725.

![Kaggle](/Image/Kaggle.png)

**However, it is worth noting that our accuracy at this point 0.7725 is not consistent with the best results on Kaggle.** The approximate accuracy of 0.780 on Kaggle indicates that the classifier we trained and used performed well in the task, but it is significantly higer than the accuracy in the training data, with a difference of up to 0.15. This led us to reflect on why there was such a noticeable gap. By comparing the final training data we used with the predicted data, we found an overlap of 394 rows of data, i.e., the data we used to train the model already contained a portion of the data that needed to be predicted. When these duplicate values were removed, the accuracy rate decreased. This is attributed to the loss of key sample information in the training set, which weakens the performance of the model.


### 3.4 Results and Discussion

The training of classification models in this part is a progressive and continuous experimentation process. Before trying the Bert model for text feature extraction, we also tried Word2Vec and TF-IDF, but the classification results were not very favorable. After constantly exploring and searching for information, we found and successfully used different Bert models and made the classification accuracy of the model improve. Finally, we performed augmentation on the original data after discovering the existence of available external data, and the accuracy proved to be improved, even without considering the overlap between the augmented training data and the test data, the accuracy of the model reached a considerable result.

Due to time constraints, we were not able to compare and tune all available models (e.g., convolutional neural networks, etc.), but only used the more common models for the practical classification task, possibly overlooking other more accurate models. On the other hand, since GridSearch is extremely time-consuming (the XGBoost classifier takes up to 2 hours to run a grid search with even a small number of parameters), and the selection and tuning of parameters is a very empirical task, we did not try every possible parameter for every classifier, which means that there is still room for improvement. Finally, more methods to improve accuracy have not yet had time to be tried (e.g., Voting Classier and other evaluation optimization methods that aggregate multiple classifiers), all of which may limit the accuracy of our final model.


## 4 Text Topic Clustering
### 4.1 Objective
This part aims at performing topic clustering and association analysis on the French text data of the test set in order to gain a deeper understanding of the features and intrinsic relationships of the text. Specifically, it aims to achieve the following objectives:
- Effectively extract text features (for nouns) from the dataset.
- Thematic classification of texts and determination of the optimal number of topics.
- Create a co-occurrence network graph of words to visually present the intrinsic relationships among high frequency words.

### 4.2 Processing
#### 4.2.1 Tokenization (Spacy)
Spacy is a powerful, efficient and fast, easy to use, and customizable NLP library. Due to the small size of the text data in this project and only perform simple text processing and feature extraction, Spacy is a lighter and faster choice.

We first defined stopwords, punctuation, and numbers to filter out common words that are not considered during text processing, such as articles, conjunctions, and common prepositions. Then, using a self-defined `spacy_tokenizer(text)` splitter function, we applied lemmatization, lowercase conversion, and stopwords and punctuation removal to the words in each document object. Finally, we obtained a list called `texts_tokenized`, which contains the text of each sentence in the training set after it has been semanticized and preprocessed. This step provides clean and normalized French text data for subsequent analysis tasks. The table following shows our outcomes.

![Token_features](/Image/Token_features.jpeg)

#### 4.2.2 Implement LDA Model
We use the Latent Dirichlet Allocation (LDA) model to cluster the topics of the text and select the best numbers.

First, we defined the `lda_model_values(num_topics, corpus, dictionary)` function. For each number of topics, a corresponding LDA model was constructed and the perplexity and coherence values were calculated. Considering the size of the sample and the effectiveness of clustering, we restricted the range of topic number to 1-20. For each of the two analysis criteria, we plotted a line graph based on the results. Perplexity is a metric used to measure the model's predictive ability for unobserved documents. Lower perplexity values indicate that the model is better at predicting unseen documents. It is observed that perplexity consistently decreases as the number of topics grows. The Coherence value helps us to evaluate the relevance and semantic consistency between topics to determine the optimal number of topics. In this case, the coherence curve shows an inflection point when the number of topics is 17. This may indicate the optimal value, i.e., further increasing the number will not bring significant performance improvement. Therefore, we choose 17 as the optimal number of topics. 

![LDA values](/Image/LDA%20values.jpeg)

Then, we built an LDA model using the optimal number of topics (17): 

```
lda = LdaModel(corpus=corpus, id2word=dictionary,num_topics=17, passes=30, random_state=1)
```

With this LDA model, we got the keywords and weights of 17 topics. Each topic can be considered as a kind of theme in a text collection, and the word weights indicate the relevance of each word to that topic. Let's take the fisrt one for example. In theme 0, "homme", ”pouvoir”, and "fille" have relatively high weight and they play an important role in that topic. Other words such as "luire" and "rvie" may be less important.

```
(0, '0.031*"«" + 0.030*"»" + 0.021*"homme" + 0.011*"pouvoir" + 0.009*"fille" + 0.009*"côté" + 0.009*"faire" + 0.009*"doute" + 0.008*"luire" + 0.008*"vie"')
```

It should be noted that in the LDA model, the topic-word distributions are derived from probability distributions, and they are not required to have a sum of probabilities equal to 1. Therefore, the absolute importance cannot be determined by the magnitude of the weights alone. Rather, weights should be considered as a measure for comparing the relative importance of different words in a topic. Thus, we analyzed the words contained in these 17 categories and their corresponding weights (probabilities), and tried to name them. 

| Topic | Name | Keywords | Topic | Name | Keywords |
|---------|---------|---------|---------|---------|---------|
| 0 | Literature | "homme", "pouvoir", "fille" | 9 | Experience and Knowledge | "pouvoir", "jamais", "entendre" |
| 1 | Technology | "robot", "bon", "compte" | 10 | Education and Society | "france", "jeune", "société" |
| 2 | Finance | "permettre", "argent", "luire" | 11 | Time and Work |  "travailler", "matin", "petit" |
| 3 | Actions and Decisions |  "main", "temps", "paris" | 12 | Language | "anglais", "changer", "suite" |
| 4 | Transportation | "voiture", "samedi", "histoire" | 13 | Family and Home | "femme", "maison", "bien" |
| 5 | Countries and Cultures | "devoir", "pays", "pourcent" | 14 | Travel | "attendre", "vacance", "marché" |
| 6 | Leisure | "ville", "passer", "manger" | 15 | Work | "travail", "arriver", "ici" |
| 7 | Exchange | "acheter", "mettre", "donner" | 16 | Language Learning | "langue", "celer", "problème" |
| 8 | Nature and Books |  "taire", "monde", "livre" |

We noticed that punctuation and inflections are still presented in the keywords corresponding to each topic. We speculate that this may be due to the absence of ‘«’, '»', and '"' in the formula `punctuations = string.punctuation +'–' + '—'`.

Finally, we use `topic = lda.get_document_topics(bow_test)` to get the probability that each sentence in the test set corresponds to each topic. Take sentence 'Vous ne pouvez pas savoir le plaisir que j'ai de recevoir cette bonne nouvelle.' for example. According to our model, it is most likely to talk about 'Actions and Decisions'. Based on our understanding, the phrase tends to express mood more than action. Therefore, it implies that our model is not very accurate. But it was still an interesting experience to explore. The table following shows our outcomes.

![TopicPredict](/Image/TopicPredict.jpeg)

#### 4.2.3 Extension: Create Network
We also wanted to use what we had learned in class about 'Graphs and Networks' to explore the connections among words. Thus, we analyzed the sentences in the test set in terms of word separation and co-occurrence to obtain the main high-frequency words (nouns) and build a relationship network graph.

First, similarly, we defined the `spacy_tokenizer_noun(text)` function, which uses the Spacy library to split the text and keep only nouns and proper names. It also removes stop words, punctuation, and digits, and lowercases and reverts each word to its original form. This leads to a lexical list `word_list`.

Then, we counted the frequency of words in the test set and returned a list of high-frequency words ranked in the top 50 (sorted by frequency in descending order).

In order to construct the co-occurrence matrix, we defined a function named `get_comatrix(text_lines_seg, topwords)`. This function takes a list of texts and a list of high-frequency words as input, counts the number of co-occurrences of high-frequency words by iterating through each text in the corpus. 

![Co-occurrences](/Image/Co-occurrences.png)

We input the obtained results into `get_net()` along with a list of high frequency words and output an undirected graph file with weights (co-occurrence counts). After we acquire the .gexf file, we import it into Gephi for further processing. Here is the social network of the top 50 high-frequency words we got from Gephi.

![Network](/Image/Network.jpeg)

From the image, we can tell that 'enfant' has the highest degree and has connection with many words. It is helpful to make us figure out and have an intuitive overview of the connection and co-occurrence of the different words.

### 4.3 Results and Discussion
This part is an extension of the project. By utilizing the text from the training data, an LDA model was trained to categorize the text into 17 distinct topics. Subsequently, the trained LDA model was applied to the test set to determine the probability of each text belonging to each topic. We extracted the top 50 most frequent nouns from the test dataset and created a co-occurrence matrix based on these high-frequency words. Using Gephi, we generated a network graph that visualizes the associations between each high-frequency word and showcases their relative importance within the network. This network graph provides an overview of the interconnections among the high-frequency words and highlights their significance in the network.

This section also has some limitations. The LDA model used is not the most advanced topic model available, and there are other research papers that have proposed more accurate models for determining text topics. When selecting the number of topics, we relied solely on coherence, and although it is generally recommended to have fewer than 10 topics, we had to choose 17 based on the coherence trend, as it achieved the highest coherence score. This higher number of topics may result in larger prediction errors in the final model. In terms of the network graph, our understanding of Gephi's advanced features is limited, so we could only create a basic network graph to display the associations between words.


## 5 YouTube Video Matching and Recommendation
### 5.1 Objective
This part aims at implementing an interactive video recommendation system. The system can suggest relevant YouTobe videos based on the user's preferences and French language difficulty preferences, helping to achieve the goal of enhancing language skills. Specifically, it is to achieve the following objectives:
- Build an interactive web page that fetches requirements and returns recommended content.
- Communicate with the YouTube API and use its features to retrieve and analyze video data.

### 5.2 Processing
#### 5.2.1 Extract Data through API
SCOPES can help us to see all YouTube data. First, we defined the `youtube_authenticate()` function to implement authentication to the YouTube API. It enables us to communicate with the YouTube API in an authorized way and use the functions provided by the API for subsequent needs such as retrieving and analyzing video data, building recommendation systems, etc. Then, we matched the videos based on the given keywords by the function `retrieve_video_list(keyword)`. Finally, we took the more specific channel title, video title, video description, and thumbnail URL information and store them in a DataFrame for further processing and analysis.

#### 5.2.2 Predict and Recommend with Interactive Widgets
This section realizes an interactive video recommendation system. We first need the user to input two parameters `keyword` and `level`, i.e. points of interest and French difficulty level, for predicting the proposed videos.

After obtaining the relevant information, we first searched the TouTube website for videos tagged as French (50 videos) related to the keywords. Then, we filtered the videos that contained French subtitles. For these videos, we extracted their subtitle text features and used the trained `etc_model_best.predict()` model to predict the difficulty of the videos. The predicted difficulty is matched with the user input level to obtain the list of videos that meet the requirements.

To improve the user-friendly design of the web page, we created an interactive window (see code in streamlit.py file) to provide users with a more convenient, intuitive, and interesting way to make recommendations. The referral page displays the title of the video, URL link, and a thumbnail of the video. Users can preview the video content and click on the link to directly access it. Taking the keyword 'merci' and the difficulty level 'C2' as an example, the following image shows the first recommended content.

![Recommendation](/Image/Recommendation.png)

### 5.3 Results and discussion

This section involves implementing additional requirements for the project, which include classifying the difficulty of French videos on YouTube using the trained classification model and creating a user interface to facilitate recommendation based on user input.

This section has several shortcomings. Firstly, due to limitations in the official YouTube API, the maximum number of search results returned is 50. After filtering the subtitles and categorizing the difficulty, only a few videos are classified as A1-B1 levels, while the majority are classified as C2. The final recommended results typically do not exceed 40. There is another reason for less number of recommendations. In determining the basis for video difficulty classification, we chose video subtitles as the input for model training. However, some videos marked by Google as French may not have French subtitles, leading to a lower quantity of displayed results. Thirdly, due to the length limitation of the Bert model's text input, Bert only considers the first 512 characters for inputs exceeding the limit. Since many subtitles are long texts, the feature extraction process may not be completely accurate, resulting in some errors in the subsequent classification. Fourthly, obtaining subtitles, performing feature extraction, and classification take a considerable amount of time. When users input their information and click the button, they typically need to wait for 30 seconds or more to receive the results, which may vary depending on computer performance. Lastly, if users perform a new search without changing the keyword but only modifying their French proficiency level, the system will rerun the search, text feature extraction, and classification steps for that keyword, rather than utilizing the results obtained from the previous search and adjusting them based on the new difficulty level. This leads to wastage of resources and time.


## 6 Conclusion 
This project explores the difficulty, topic, and core vocabulary of French texts (sentences) using machine learning methods from a personalized language learning perspective. We have successfully built an interactive page that can recommend French videos based on topic and difficulty. Overall, we have generally accomplished our initial vision and goals for the project.

During the project, we actively learned and practiced various models for tokenizatio, feature extraction, and classification, and gained a deeper understanding of the features, capabilities, and limitations of these classifiers. We learned to evaluate the performance of models from a more detailed perspective (sample balance, accuracy balance) by discussing the accuracy, precision, and recall of results from various aspects. At the same time, we also benefited from actively linking what we learned in class, such as Graph Analytics, to the project and conducted a lot of extended studies.

The most straightforward feeling from this project is the complexity and challenge of machine learning. From data pre-processing and feature engineering to model training and evaluation, each step involved a large amount of knowledge and techniques. At the same time, this project also showed us the great potential of machine learning in practical applications. Through machine learning algorithms, we are able to discover patterns and regularities from massive amounts of data, enabling automated decision making and prediction. However, it also made us aware of the shortcomings of our own capabilities. There are still many ideas that we have not realized due to time and capacity constraints. We should keep our passion for learning and continuously expand our knowledge and skills to exploit the greater possibilities of machine learning.

Last but not least, it was a very enjoyable journey of discovery and good teamwork. The two of us discussed and logically organized each part intensively. Also, we exchanged with other groups to make progress together.