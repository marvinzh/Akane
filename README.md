## Akane
### a collection of Machine Learning algorithms for Regression, Classification, and Clustering

### Introduction
Akane is a collection of Machine Learning algorithms for Regression, Classification, and Clustering. Currently, Akane provides the following algorithms:
- **Regression**
    - Simple Linear Regression
    - Linear Regression with L1 Regularization (a.k.a. Lasso Regression)
    - Linear Regression with L2 Regularization (a.k.a. Ridge Regression)
    - Linear Regression with L1 & L2 Regularization (a.k.a. Elastic Net)
    - Polynomial Regression
    - Kernel Regression

- **Classification**
	- Logistic Regression
	- Decision Tree
	- AdaBoost
	- Neural Networks

- **Cluster**
	- k-means
	- nearest neighbor
	- gaussion mixure model(trained by EM)

I implemented these algorithms from scratch. currently, there are still lots of works to be done (e.g. advanced model, test unit, dev documentation, examples, and API documentation), and the struct of this software are not good. Therefore, I'm planning spend few years woking on it. There is a TODO list in the below I made to show you what I'm doing now. after finish the test unit module, I will refactoring this software. 

### Prerequisite
- Python(3.5.0)+
- numpy(1.11.0)+
- scipy(0.17.1)+
- pandas(0.18.1)+
- PyLBFGS(0.2.0.3)+

### Todo List
- EM Implementation
- KD-tree and Locality Sensitive Hashing(LSH) implementation (for nearest neighbour)
- Support Vector Machine Implementation
- examples
- development notes
- test unit
- API documentation


### Update notes
- 2016-11-16 add k-means algorithm
