# COVID-19_cases_and_mobility_predictor
The prediction of new cases and mobility from COVID-19 mobility and new cases datasets, using decision tree algorithms like CART and C4.5.
All code is written in Python.

# Decision trees
A decision tree is a predictive model that uses a flowchart-like structure to make decisions based on input data. It divides data into branches and assigns outcomes to leaf nodes. They are used for classification and regression tasks, thus providing easy-to-understand models.

# CART Algorithm
  The CART algorithm is a type of classification algorithm that is required to build a decision tree on the basis of Gini’s impurity index. Nodes are split into subnodes on 
  the basis of a threshold value of an attribute. The CART algorithm does that by searching for the best homogeneity for the subnodes, with the help of the Gini Index 
  criterion.
  The formula for Gini impurity is given as follows:
  
  <img width="187" alt="image" src="https://github.com/snehita1212/COVID-19_cases_and_mobility_predictor/assets/92868475/12c34309-029d-47f4-898b-b32910a40baa">
  where pi is the probability of an object being classified to a particular class.

# C4.5 Algorithm
  C4.5 algorithm is an improvement over the ID-3 algorithm. The splitting criterion used by C4.5 is the normalized information gain (difference in entropy). The attribute with 
  the highest normalized information gain is chosen to split nodes. The C4.5 algorithm then recurses on the partitioned sub-lists.
  The formula for information gain is given as follows:

  <img width="409" alt="image" src="https://github.com/snehita1212/COVID-19_cases_and_mobility_predictor/assets/92868475/fed79d9e-b877-4bb7-ad93-edaa67d39bc9">
  where weighted entropy of all classes is removed from the total entropy.
