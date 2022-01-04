# DDoS-Detection

With the increase in technological advancement, especially the internet, there come various kinds of network attacks. DOS attacks pose one of the most challenging security threats in today’s generation of internet. In addition, the Distributed Denial of Service (DDoS) attacks are of a particular concern whose impact can be proportionally severe. The challenging part of this attack is that it comes with little or no advance warning and could easily exhaust away the communication or computation resources of the victim in a short span of time. These attacks are becoming advanced day-by-day and are increasing in number thus making it difficult to detect and counter such attacks. In addition to detecting the upsurge of packets during DDoS attack using Wireshark, we would be incorporating Machine Learning techniques for effective detection of DDoS flooding attack such as K-Nearest Neighbors, SGD, Multi-layer Perceptron, Logistic Regression, Naive Bayes, Support Vector Machine, DNN, XGBoost, Decision Tree, Quadratic discriminant etc. A detailed comparative analysis of the aforementioned algorithms is performed and is evaluated based on the accuracy metrics. 



## Details of algorithms used

#### Naïve Bayes Classifier
Naïve Bayes classifier is a simple probabilistic ML model that calculates the probabilities for each class in a dataset and adopts discriminative learning to predict values of the new class. It is based on Bayes theorem; assuming the features to be independent, we can find the probability of A (hypothesis) happening given that B (evidence) has occurred. It is naïve as the presence of one predictor/feature does not affect the other.

#### Decision Tree
Decision Tree belong to the class of non-parametric supervised learning method. It is mainly used for the purpose of solving the regression and the classification problems. The major aim is to build a model which predicts the value of a target variable which is done by learning the simple decision tree rules that are inferred from the features of the data. The tree is seen as a piecewise constant approximation. For solving the classification problem, the class DecisionTreeClassifier is used. The class is well equipped to perform a multi-class classification on the dataset. The classifier takes two arrays as input: an array X or dense, having shape (n_samples, n_features) which hold the training samples of the dataset, and an array Y which have integer values having shape (n_samples) which hold the class labels of the corresponding training samples. After model 
fitting, the model is used for making predictions of class of the samples. In the case of multiple classes with the exact same and highest probability, the classifier 
tends to predict the class which has the lowest index among the classes. Besides outputting to a specific class, the probability of each class could be predicted 
which constitutes the fraction of the training sample of the class in a leaf. The classifier is capable for classifying both multi-class classification and binary lassification.

#### Deep Neural Networks
One of the most well-known and recent models is the Deep Neural network which can be considered as a stack of neural networks i.e., a network composed of several layers. DNN has been successfully applied in several applications, including regression, classification, or time series prediction problems using simple auto-regressive models. The architecture comprises of at least 3 layers of nodes namely input layer, hidden layer and output layer which are interconnected; the flow of data takes place via one direction from input nodes to output node. Further DNN uses backpropagation as the training algorithm and activation function (usually sigmoid) for classification process. We train a deep neural network to classify normal and DDoS attack states by using a carefully chosen set of network statistics as an input signal.

#### Stochastic Gradient Descent Classifier
This Classifier implements regularized linear models such as SVM, logistic regression, etc. by making use of Stochastic gradient descent (SGD) optimization technique in the training process. The gradient of the loss for each sample is calculated by this optimizer at a time and the model is updated by estimating minimum cost function which is obtained with a decreasing learning rate or strength schedule. SGD Classifier is an efficient estimator for large scale problems as it allows minibatch learning via the partial fit method. Simple linear classifiers don’t work if the records cannot be kept in RAM, however SGD classifier continues to work. This model is sensitive to feature scaling and require fine tuning of many hyperparameters such as number of iterations and regularization parameter for a good performance.

#### K-Nearest Neighbors
K-Nearest Neighbor (K-NN) is one of the simplest Supervised Machine Learning algorithms which presumes the similarity between existing data and new data and put the new case into the category that is most like the available ones. It classifies a new data point based on the similarity of stored available data i.e., when any new data appears then it can be easily classified into a well-suited category by using K- NN algorithm. The KNN classifier has the ability to effectively detect invasive attacks as well as achieve a low fall-out ratio. It can distinguish between the normal and abnormal behavior of the system and is used to classify the status of networks to each phase of DDoS attack.


#### Support Vector Machine
Support Vector Machines (SVM) is one of the most favored ML algorithms for many applications, such as pattern recognition, spam filtering and intrusion detection. There are several SVM formulations for regression, classification, and distribution estimation. It is derived from linearly separable and the most optimal classification hyperplane. There is a training set D = {(X1, y1), (X2, y2) …. (Xn, yn)}, where Xi is the characteristic vector of the training sample and yi is the associated class label. takes +1 or −1 (y belongs to {+1, -1}) indicating that the vector belongs to this class or not. It is said to be linearly separable if there exists a linear function that can separate the two categories completely; otherwise, it is nonlinearly separable. As DDoS attack detection is equivalent to that of a binary classification problem, we can use the characteristics of SVM algorithm collect data to extract the characteristic values to train, find the optimal classification hyperplane between the legitimate traffic and DDoS attack traffic, and then use the test data to test our model and get the classification results.

#### Random Forest
The Random Forest classifier makes use of ensemble learning technique as it constitutes of many decision trees. All the individual trees present as a part of random forest provide a class prediction. Subsequently, the class with the highest number of votes becomes the prediction of the entire model. The core idea of the classifier is to have a significant number of trees which operate together as a whole to outperform any of the individual constituent models. The key is low correlation between the models. Uncorrelated models have the capability to produce more accurate models than any of the individual predictions. The main reason is that the trees protect one another from individual errors. While some trees might be wrong, if many other trees are right, then, the group of trees would be able to move towards the right direction. The classifier makes use of feature randomness and bagging to build each individual tree to create an uncorrelated forest of trees.

#### XGBoost Classifier
XGboost is a classifier which is based on the decision-tree-based ensemble machine learning. It makes use of a gradient boosting framework. For the prediction of unstructured data such as images, text etc. the artificial neural network tends to perform better as compared to the other frameworks or algorithms. However, decision tree-based algorithms are considered to be the best when it comes to the small-to-medium structured/tabular data. The algorithm is based on the base GBM framework by algorithmic enhancements and system optimizations. In other words, it is an optimized gradient boosting algorithm which makes use of tree pruning, parallel processing, tree pruning and handling of the missing values and makes use of regularization in order to avoid bias and overfitting.

#### Quadratic Discriminant Analysis
In Quadratic Discriminant Analysis (QDA), each class follows a Gaussian distribution and is generative. It is very much like that of Linear Discriminant Analysis with the exception that the covariance and mean of all the classes are equal. The class specific prior refers to the proportion of the data points which belong to that class. The class specific covariance matrix refers to the covariance of the vectors which belong to that class. The class specific mean vector refers to the average of the input variables which belong to that class.



## Dataset Description
The DDoS attack dataset is a SDN specific dataset that is generated by making use of the mininet emulator and is mainly used for the classification of traffic by numerous deep learning and machine learning algorithms. The process involved for the creation of the dataset includes the creation of ten topologies in mininet where the switches were connected to a single Ryu controller. The network simulation runs for the both the benign UDP, ICMP and TCP traffic as well as the collection of malicious traffic for TCP Syn attack, ICMP attack and UDP flood attack. The dataset includes 23 features in total where some of the data is extracted from the switches and others were calculated. Extracted features which are present in the dataset include: -

Packet_count – refers to the count of packets <br />
byte_count – refers to the count of bytes in the packet <br />
Switch-id – ID of the switch <br />
duration_sec – packet transmission (in seconds) <br />
duration_nsec – packet transmission (in nanoseconds) <br />
Source IP – IP address of the source machine <br />
Destination IP – IP address of the destination machine <br />
Port Number – Port number of the application <br />
tx_bytes – number of bytes transferred from the switch port <br />
rx_bytes – number of bytes received on the switch port <br />
dt field – shows the date and time which has been converted into a number and the flow is monitored at a monitoring interval of 30 seconds<br />
<br />
The calculated features present in the dataset include: <br />
Byte Per Flow – byte count during a single flow<br />
Packet Per Flow – packet count during a single flow<br />
Packet Rate – number of packets transmitted per second and calculated by dividing the packet per flow by monitoring interval<br />
number of Packet_ins messages – messages that are generated by the switch and is sent to the controller <br />
Flow entries of switch – entries in the flow table of a switch which is used to match and process packets<br />
tx_kbps – Speed of packet transmission (in kbps)<br />
rx_kbps - Speed of packet reception (in kbps)<br />
Port Bandwidth – Sum of tx_kbps and rx_kbps<br />

The output feature is the last column of the dataset i.e. class label which classifies the traffic type to be benign or malicious. The malicious traffic is labelled as 1 and the benign traffic is labelled as 0. The simulation of the network was run for approximately 250 minutes and 1,04,345 instances of data were collected and recorded. Further, the simulation was run for a given interval to collect more instances of data.

Link for dataset: https://data.mendeley.com/datasets/jxpfjc64kr/1

## Results
DDoS attacks analysis and detection were performed using machine learning method. In this work, a SDN specific dataset is used. The dataset originally includes 23 features. The output feature is the last column of the dataset i.e. class label which classifies the traffic type to be benign or malicious. The malicious traffic is labelled as 1 and the benign traffic is labelled as 0. It has 104345 instances. The null values were observed in the rx_kbps and tot_kbps and were hence dropped for model development. The data processing steps were completed, including data preparation/cleaning, One Hot encoding, and normalization. After one hot encoding the dataframe had 103839 instances with 57 features and was fed into the model.A Deep Neural Network was used as the proposed model. The efficacy of our proposed model was observed to be higher than that of the baseline classifiers used. The accuracy of our proposed model was observed to be 99.38% which is approximately 1.21% higher than the next best model XGBoost whose accuracy stands at 98.17%
