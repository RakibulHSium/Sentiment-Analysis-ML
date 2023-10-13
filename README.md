
# Sentiment Analysis by Machine Learning Algorithms

## 1. Abstract
The principles’ objective of sentiment analysis is to extract views, attitudes, and emotions from social media platforms such as Twitter. It's a hot topic in academic circles. Conventional sentiment analysis focuses mostly on textual data. On Twitter, users send their updates on various subjects in the form of tweets or status, which is the most popular microblogging online networking site. An extensive series of Advance Data pre-processing methods And Machine Learning Algorithms proposed method is outlined in this study, which utilizes two publicly accessible labeled datasets (Twitter Dataset and Vaccination Dataset) in order to make the tweeted messages more easily understandable by standard language handling algorithms. In the Advance Data pre-processing phase author has used several methods to cleaning the data which is Transform Cases, Tokenize, Filter Stop Words, Filter words by length, stemming and for the text manipulation author has used NLTK(Natural Language Toolkit). After Advance pre-processing phase author used TF-IDF Vectorization. As a result, supervised machine learning algorithms and TF-IDF is used in this phase. In addition, on the second phase Machine learning classifiers like Naïve Bayes, Logistic Regression, Support Vector Machines, Random Forest and XGBoost-based sentiment analysis models are proposed for training Advance pre-processing data and have done two experiments on Twitter Dataset and Vaccination Dataset. Ultimately, the main goal is to make it easier and more readable to understand people's feelings. Twitter sentiment analysis divides tweets into two categories depending on their content: positive sentiment and negative sentiment. Machine learning classifiers has done a great job to do experiments on selected datasets. There are a variety of uses for such classifiers that include assessing the feelings of businesses, political parties, as well as analysts and others. After the experiment the tweets are appropriately classified using machine learning methods trained on training data. Also, I analyzed some deep learning approaches on sentiment analysis from previous work by different researchers with their proposed method, which significantly improved performance metrics, with an increase of 20% in accuracy and 10% to 12% in precision but only 12–13% in recall as compared to current approaches. In the datasets tweets are demonstrates by positive, neutral, and negative sentiments were found to account for 45%, 30 %, and 25% of all tweets, respectively. This shows that deep learning has more effective ways and is feasible for sentiment classification in Twitter data.


## Keywords
**Sentiment, Sentiment Analysis,Twitter, Data mining ,NLTK, Classification, Naïve Bayes, SVM,Linear SVC, XGBoost, Logistic Regression, Supervised Learning, Machine Learning**
## Research Problem
To begin, we uncovered several research relating to sentiment analysis on Twitter data sets, with Naive Bayes and Random Forest being the most often used algorithms for sentiment prediction. Nowadays, XGBoost and CNN- LSTM are popular machine learning algorithms; these algorithms aid in producing better results with less computer resources and quicker iterations. There is no relevant paper on predicting sentiment on a twitter dataset using XGBoost and CNN-LSTM in this search. Thus, the performance of XG Boost and CNN-LSTM is compared to that of Naive Bayes and Random Forest in order to determine the best-fit algorithm for sentiment analysis of Twitter data. The findings of this study contribute to the improvement of prediction performance, the reduction of temporal complexity associated with data prediction, and the enhancement of computational complexity and explain ability. This study is beneficial to companies since it allows them to assess the product and improve their forecast performance.

## Motivation and Objectives
The primary goal of this study is to compare the performance of machine learning algorithms such as Naïve Bayes, Random Forest, XGBoost, and deep learning algorithms such as CNN-LSTM model on Twitter datasets based on the size of the datasets being analyzed.





## 2. Literature Review

(1). Pang and Lee classified sentiments using the polarity dataset. They classified the reviews subjectively and objectively. They took into account just the subjective element, since the objective section has no information regarding the mood. They used the minimum-cut formulation in the graph technique to extract the subjective section of the text for review from the complete text. They classified reviews using SVM and NB classifiers in conjunction with a minimal cut formulation.

(2). Salvetti et al., introduced the idea of overall opinion polarity (OvOp) in the context of review categorization using machine learning techniques. They classified using Naive Bayes and Markov Model methods. In this study, wordnet supplied the hypernyms and the Part of Speech (POS) tag served as the lexical filter for classification. They assert that the result produced by the wordnet filter is less precise than the result produced by the POS filter. 

(3). Beineke et al., used a Naïve Bayes method to classify attitudes. For sentiment prediction, they extracted a pair of linearly combinable derived features. They introduced more derived features and evaluated the relative impact using labeled data to improve the model's accuracy. Additionally, they examined anchor words, which are words with many meanings. They analyzed five pairs of positive and five pairs of negative anchor words, yielding a total of 25 possible pairings for study. They followed Turney's approach, which entails creating a new corpus of label documents from an existing one .

(4). Mullen and Collier used the SVM technique to do sentiment analysis, which involves assigning values to a small number of chosen phrases and then combining them to create a classification model. Additionally, favorable values are provided to several groups of attributes that are relevant to the issue, assisting with categorization. The authors conducted a comparison of their suggested technique to data annotation using topic annotation and manual annotation. Their suggested technique outperformed topic annotation in terms of outcomes, however results when compared to manually annotated data need additional refinement.

(5). Zhang et al., suggested a rule-based categorization system for reviews . Their technique is divided into two stages: sentence sentiment analysis and document sentiment aggregation. They break down the text into component phrases and give polarity to each one. The polarity ratings of all sentences are then added together to determine the text's overall polarity. They analyzed the Euthanasia dataset, which comprises 851 Chinese articles, and the AmazonCN dataset, which contains 458,522 reviews across six categories, including books, music, movies, electrical appliances, digital products, and photography. They used SVM, neural networks, and decision trees to categorize the reviews.

(6). Yessenalina et al., presented a collaborative two-level technique for sentiment categorization at the document level . Their technique takes subjective statements from the text and classifies the document based on these phrases. Their training strategy treats each phrase as a hidden variable and collectively learns to predict the document label that prevents inaccurate sentence labels from propagating. To maximize document-level accuracy, their model solves the sentence extraction subtask just to the degree necessary to categorize the document sentiment effectively. They classified movie reviews and US Congressional floor debates using the SVM machine learning approach.


(7). Kanayama and Nasukawa suggested a Japanese-language implementation of domain-oriented sentiment analysis. The suggested technique picks polarity clauses that express the domain's goodness and badness. They employed an unlabeled dataset for lexicon analysis and assumed that polar sentences with the same polarity appeared sequentially until the context changed due to adversative expressions. They gathered candidate polar atoms and their likely polarities using this technique. Additionally, they addressed the inter- and intra-sentential contexts in order to produce more polar atoms. Additionally, they discovered coherence precision and coherent dependence, which aid in the analysis of documents in new domains. They used unsupervised sentiment analysis to gather Japanese corpora from discussion boards covering four distinct areas, namely digital cameras, movies, mobile phones, and automobiles.


(8). Wan used unsupervised sentiment analysis on Chinese reviews due to the difficulty of obtaining tagged reviews for study . He used Google Translate, Yahoo's Fish, and Baseline translation to convert the Chinese reviews to English ones. He then utilized ensemble techniques to enhance the analysis result by combining the individual analysis findings for both languages. He performed unsupervised sentiment analysis using six distinct ensemble approaches, including average, weighted average, max, min, average of max and min, and majority voting. He gathered 1,000 product reviews from IT168, a major Chinese IT product evaluation website.


(9). Zagibalov and Carroll suggested an unsupervised sentiment classification method based on autonomous seed task selection . They began by analyzing a single human-selected word, 'good,' and then extracted a training sub- corpus using an iterative process. They used the terms "lexical item" and "zones" to refer to any sequence of Chinese characters and "zones" to refer to a series of characters terminated by punctuation marks. Following that, each zone is categorized into several polarity groups according to the predominance of polarity vocabulary words. To determine the polarity of a document, the difference between the positive and negative zones is calculated, and if the difference is positive, the document is categorized as positive; otherwise, it is labeled as negative. They classified the product reviews acquired from the IT168 website, which total 29531 reviews after deleting duplicates.

(10). Goldberg and Zhu suggested a graph-based semi-supervised learning strategy for addressing the rating interference sentiment analysis challenge .They built graphs using both labeled and unlabeled data in order to express particular task assumptions. They then used optimization to produce a smooth rating function for the whole graph. They thought that the measure of similarity between two papers should be larger than zero. They conducted the experiment using positive sentence percentages and mutual information-modified cosine word vector similarities. They evaluated the movie reviews documents associated with four distinct class labels provided in the Cornell digital library's "Scale dataset v1.0" .


(11). Sindhwani and Melville introduced a semi-supervised sentiment prediction system that makes use of unlabeled instances and lexical prior knowledge .Their technique is based on a bipartite graph representation of the data and a cooperative sentiment analysis of documents and words. They have added sentimental keywords into their analytical paradigm. They leveraged a vast quantity of unsupervised data in order to adapt to a new domain with minimum supervision. They analyzed the movie reviews dataset proposed by Pang and Lee , as well as two additional blog datasets. They generated a dataset for sentiment analysis that includes information regarding the IBM Lotus brand. The second batch of blogs has 16742 political blogs.


(12). Melville et al., proposed a unified approach that makes use of background lexicon information in terms of word-class association and refines it for each domain .They developed a generative model based on a lexicon of sentiment-laden phrases, as well as a model for training on the label dataset. To collect information, these two models are adaptively pooled to construct a composite multinomial NB classifier. They utilized the tagged document to fine-tune the information gathering, which is based on a domain-neutral vocabulary. They classified 20488 technical blogs comprising 1.7 million posts from IBM Lotus collaboration software, 16,741 political blogs containing two million posts, and the Pang and Lee movie review dataset.
## 3. Methodology
### 3.1 Setting Up for Software Environment

The following python libraries are used to create the machine learning models in this experiment:
**NLTK:** It's a Python module that works with human language data and gives you a simple way to access lexical resources like WordNet and text processing tools. Classification, tokenization, stemming, tagging, parsing, and semantic reasoning are all done using these lexical resources.

**Pandas:** It's a Python library that serves as a data analysis tool and works with data structures. Pandas performs the whole data analysis pipeline in Python, eliminating the need to transition to a more domain-specific language such as R.

**Tweepy:** It is used to make a connection to the Twitter API and to collect tweets from Twitter. This module is used to stream real-time tweets straight from Twitter.

**NumPy:** NumPy is the fundamental package for computing with Python. It is used to add support to multi- dimensional arrays and matrices, with a large collection of high-level mathematical functions.

**scikit-learn:** It's a straightforward and effective data mining and data analysis tool .

**matplotlib:** Plots, histograms, power spectra, bar charts, and other graphics may be generated using this Python package. The measurements are plotted using the matplotlib.pyplot package.

**Gensim:** It's used to mechanically extract as many semantic topics as possible from texts. Gensim is an unstructured text data processing tool. The algorithms in Gensim, such as Word2Vec, look at statistical co- occurrence patterns in a corpus of training texts to discover the semantic structure of a phrase automatically. These are algorithms that are not supervised. After these statistical patterns are found, any plain text materials may be easily represented in the new, semantic form and analyzed for theme similarity against other articles .

**Keras:** Keras is a Python-based high-level neural network API that may be used with TensorFlow, CNTK, or Theano. It was created with the goal of allowing for quick experimentation. To undertake successful research, you must be able to get from concept to outcome as quickly as feasible .
The methods used in this experiment were imported from sklearn, which allows us to easily train and test classifiers on datasets. The following sklearn tools are utilized in this project:
1. Naive Bayes
2. Logistic Regression
3. Random Tree
4. XGBoost
5. Support Vector Machine

### 3.2 Data Preparation 
#### Data collection
Collecting relevant datasets regarding a topic of interest is what data collecting entails. For the specified time period of study, the tweets are gathered using Twitter's streaming API, or any other mining tool (for example, WEKA). The format of the obtained text is changed to suit your needs (for example JSON in case of the dataset gathered is critical to the model's efficiency. The model's efficiency is also determined by how the dataset is divided into training and testing sets. The training set is the most important factor in determining the outcome. This research aims to tackle the challenge of text sentiment analysis on twitter text data by predicting whether or not a tweet is racist or sexist. The information was obtained from Analytics Vidhya.

### 3.3 Feature Extraction
To successfully use machine learning algorithms to Sentiment Analysis datasets, it is critical to extract confident characteristics from the textual content that result in proper categorization. Typically, the original textual content data is supplied as a component of a Feature Set (FS), where FS = (feature 1, feature 2 . . . feature n). Two FE techniques are used in this study: the Term Frequency–Inverse Document Frequency (TF–IDF) and the N-gram. 

**Term Frequency–Inverse Document Frequency Algorithm**
The Term Frequency–Inverse Document Frequency (TF–IDF) method of classifying textual material is frequently utilized. The TF–IDF model is made up of two parts: term frequency and inverse document frequency. "The fundamental monitoring of the frequency with which a specific comment occurs in a given text" is how the term "frequency" is defined. To calculate the inverse document frequency, divide the total number of documents by the total number of documents that contain a certain term. The output score is the sum of the most commonly appearing phrases in texts and the least frequently occurring keywords in each document when both components are combined. This aids in the identification of important and long portions within a text .

**N-Gram Algorithm**
N-grams are widely employed in NLP applications and are useful for capturing textual context to some extent. It's debatable if using a higher level of n-gram is useful or not. Many academics claim that the unigram outperforms the bigram in identifying movie reviews by sentiment polarity, although other analysts and researchers discovered that bigrams and trigrams beat unigrams in various reviews datasets.

## Model Frameworks



![Model](https://raw.githubusercontent.com/RakibulHSium/Sentiment-Analysis-ML/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/TwitterSentiment/Result/model.png)

*Flowchart: Proposed methodology of Sentiment Analysis by using Machine Learning Algorithms*


## 4. Result and Analysis
### 4.1 Ex1: Sentiment Analysis by using Machine Learning Algorithms(Twitter Dataset)
**I. Data Advance Preprocessing**
In the train datasets we have 2242 tweets labeled as non-racist/sexist. So, it is and imbalanced classification challenge. Now
we will check the distribution of length of the tweets, in terms of words, in both train and test data.

![tweet length](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/TwitterSentiment/Result/tweetlength.png)

*Figure 4.1: The tweet-length distribution is more or less the same in both train and test data.*

**II. Understanding the impact of Hashtags on tweets sentiment**

[![Non Racist](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/TwitterSentiment/Result/Image5.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)

*Figure 4.2: Most frequent hashtags appearing in the non-racist/sexist tweets*


All these hashtags are positive and it makes sense. I am expecting negative terms in the plot of the second list. Let’s check the most frequent hashtags appearing in the racist/sexist tweets.


[![Racist/sexist](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/TwitterSentiment/Result/image6.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)

*Figure 4.3: Most frequent hashtags appearing in the racist/sexist tweets*

[![cmatrix-naive](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/TwitterSentiment/Result/image7.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)

### 4.2 Ex2: Covid-19 Vaccination Sentiment Analysis by Machine Learning Algorithms(Vaccination Datasets)

**I. Data Advance Preprocessing**


[![image1](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/VaccinationDataSentiment/Result/image1.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)

[![image 2](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/VaccinationDataSentiment/Result/image2.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)


*Figure 4.4: Vaccination Dataset Text Polarity with sentiment*

**II. Vaccination Tweets Sentiment Visualization**

[![img 3](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/VaccinationDataSentiment/Result/Image3.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)

*Figure 4.5: Tweet visualization by count from vaccination datasets*


**III. Understanding the common words used in the tweets : WordCloud**

Now, I want to evaluate the feelings' distribution throughout the train dataset. Understanding the frequent terms by producing WordCloud is one technique to achieve this objective. A WordCloud is a graphical representation in which the most common words are shown in big font size and the less frequent words in smaller font size. Let's see all the words in our data using a WordCloud.

[![img5](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/VaccinationDataSentiment/Result/image5.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)

*Figure 4.6: Most Frequent words in positive tweets*

[![img 6](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/VaccinationDataSentiment/Result/image6.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)

*Figure 4.7: Most frequent words in Negative tweets*

[![img7](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/VaccinationDataSentiment/Result/image7.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)

*Figure 4.8: Most frequent words in Neutral tweets*


**Table 4.1: Classification Accuracy of Logistic Regression**

|       | Precision(%) | Recall(%) | F1 Score(%) | Support |
|-------|--------------|-----------|-------------|---------|
| Negative | 86% | 32% | 46% | 226 |
| Neutral | 79% | 99% | 88% | 1021 |
| Positive | 94% | 82% | 87% | 862 |
| Accuracy | | | 85% | 2109 |
| Macro avg | 86% | 71% | 74% | 2109 |
| Weighted avg | 86% | 85% | 83% | 2109 |


[![cm-log](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/VaccinationDataSentiment/Result/image10.png)](https://github.com/RakibulHSium/Sentiment-Analysis-ML)


*Figure 4.9: Plotting Result of Confusion Matrix of Logistic Regression*


|         | Precision(%) | Recall(%) | F1 Score(%) | Support |
|---------|--------------|-----------|-------------|---------|
| Negative | 83% | 45% | 58% | 226 |
| Neutral | 83% | 99% | 90% | 1021 |
| Positive | 95% | 85% | 90% | 862 |
| Accuracy | | | 87% | 2109 |
| Macro avg | 87% | 76% | 79% | 2109 |
| Weighted avg | 88% | 87% | 87% | 2109 |

**Table 4.2: Classification Accuracy of Support Vector Machine**


![svm p20](https://github.com/RakibulHSium/Sentiment-Analysis-ML/raw/4f9ff3b86d4d52511110fedd2c7e266bdbdc1588/VaccinationDataSentiment/Result/image11.png)

*Figure 4.10: Confusion Matrix of Support Vector Machine Where Parameter c=20*

## Comparison between machine learning classifiers accuracy for both selected dataset 

| Model              | Naïve Bayes | Logistic Regression | Support Vector Machine | Random Forest | XGBoost |
|--------------------|------------|----------------------|------------------------|--------------|---------|
| Twitter Dataset    | 94%        | 66.1%                | 65.4%                  | 54.9%        | 69.8%   |
| Vaccination Dataset |            | 85%                  | 87%                    |              |         |









