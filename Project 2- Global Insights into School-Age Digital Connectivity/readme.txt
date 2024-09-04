I have first converted the excel file into a CSV file manually.
In order to run the code, please make sure that the path of the two datasets(training and testing) are correctly loaded.
Please copy the correct file name while loading the data in the jupyter notebook.
 
Please know that I have combined the 2 data sets given to us into one, so the results are for one dataset. The training and testing is done seperately within the dataset.

Exploration of Datasets and Model Development

This Jupyter Notebook shows how to use the pandas library to load and examine two datasets. It also demonstrates how to make various graphs for data visualisation using the matplotlib and seaborn libraries. The notebook also contains training of the datasets using K-Nearest Neighbours (KNN) and Decision Tree classifier, two machine learning models.


Requirements

Jupyter Notebook - Python 
- seaborn, pandas, matplotlib, scikit-learn 

Installation ##

1. Download the notebook file ('Assignment2.ipynb') to your local computer or clone the repository.
2. Run the following command in your terminal to install the necessary libraries:
---------pip set up pandas Scikit-Learn, Seaborn, and Matplotlib


Usage

1. Start Jupyter Notebook on your computer locally.
2. Click the 'Global Insights into School-Age Digital Connectivity.ipynb' file to open it.
3. To run the code and see the results, go through each cell in the notebook in order.

Dataset

The notebook is predicated on the existence of two CSV datasets, Training.csv and Testing.csv. Before executing the notebook, make sure that these files are located in the same directory.


Book Page Contents

The following sections make up the notebook:

1. Dataset Loading: The datasets are loaded into distinct dataframes using the pandas package.
2. Data Exploration: Several exploratory data analysis methods are used, such as summary statistics, data visualisation with graphs (using the programmes Matplotlib and Seaborn), and any extra analysis considered required.
The datasets are integrated into one dataset for model training in step three. Then, using the training sets and testing sets, the KNN and Decision Tree classifiers are trained.
4. Results and Discussion: The trained models' performance metrics are shown, along with any findings or conclusions that can be drawn from the study.
