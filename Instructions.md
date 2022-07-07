Regularized Linear Regression Project Tutorial
Edit on Github

    This project dataset has a lot of features related to socio demographic and health resources data by county in the United States, right before the Covid-19 pandemic started (data from 2018 and 2019).

    We want to discover if there is any relationship between health resouces and socio demographic data. Choose one target variable (related to health resources), and use the LASSO model to reduce features to the most important ones for your target.

    Find the parameters for your linear regression between your selected features and your chosen target.

🌱 How to start this project

You will not be forking this time, please take some time to read this instructions:

    Create a new repository based on machine learning project by clicking here.
    Open the recently created repostiroy on Gitpod by using the Gitpod button extension.
    Once Gitpod VSCode has finished opening you start your project following the Instructions below.

🚛 How to deliver this project

Once you are finished creating your model, make sure to commit your changes, push to your repository and go to 4Geeks.com to upload the repository link.
📝 Instructions

U.S.A. county level sociodemographic and health resource data (2018-2019)

There is a 'data-dictionary' file in this folder that explains the meaning of each feature. You need to select one of the features related to health resources as your target variable and then use the LASSO regression to discover which features are the most important as factors to explain your target variable.

Step 1:

The dataset can be found in this project folder as 'dataset.csv' file. You are welcome to load it directly from the link (https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv), or to download it and add it to your data/raw folder. In that case, don't forget to add the data folder to the .gitignore file.

Time to work on it!

Step 2:

Use the explore.ipynb notebook to find correlations between features or between feature and your chosen target.

Don't forget to write your observations.

    Consider doing feature scaling before applying LASSO.

Step 3:

Now that you have a better knowledge of the data, apply the LASSO model which already includes feature selection to obtain the most important features that influence in your target variable.

    We are not going to predict anything, but don't forget to drop all the features related to health resources from your X (features) dataset, and define your chosen target as your 'y'.

Use ordinary least squares regression to choose the parameters that minimize the error of a linear function.

Step 4:

Use the app.py to create your pipeline that selects the most important features.

Save your final model in the 'models' folder.

In your README file write a brief summary.