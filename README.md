# Flight-Price-Prediction-System
This project involves predicting flight prices based on various factors such as the airline, origin, destination, stops, and travel date.
## **Project Overview**:

### **Steps**:

- **Data Cleaning and Preprocessing**:
    - **Remove Unnecessary Columns**:
        - Drop empty or irrelevant columns such as `Unnamed: 11` and `Unnamed: 12`.
    - **Convert Date Columns**:
        - Convert date-related columns into proper datetime formats and extract meaningful features like **day**, **month**, and **year**.
    - **Transform Categorical Variables**:
        - Convert categorical columns (such as **airline**, **from**, **to**, and **stops**) into numerical representations using techniques like **Label Encoding** or **One-Hot Encoding**.
    - **Convert Duration**:
        - Convert the `duration` column from text format (e.g., '3h 15m') to a numerical value (e.g., total minutes).
    - **Clean Price Column**:
        - Clean and convert the `price` column from text format (e.g., '₹4500') into a numerical value (e.g., 4500).
- **Exploratory Data Analysis (EDA)**:
    - **Price Distribution**:
        - Analyze the distribution of flight prices across different airlines, routes, and travel times. Identify any significant patterns in pricing.
    - **Visualize Relationships**:
        - Visualize the relationship between **flight duration**, **departure time**, and **price** using scatter plots or line charts.
    - **Impact of Stops on Price**:
        - Investigate how the number of stops (non-stop, 1 stop, 2 stops, etc.) affects the flight price using bar charts or box plots.
- **Model Selection and Training**:
    - **Train-Test Split**:
        - Split the dataset into training and test sets using **train_test_split**.
    - **Regression Algorithms**:
        - Use regression algorithms such as **Linear Regression**, **Random Forest Regressor**, or **Gradient Boosting Regressor** to train the model.
    - **Model Evaluation**:
        - Evaluate the model using appropriate metrics like **Mean Absolute Error (MAE)** or **Root Mean Squared Error (RMSE)** to assess the accuracy of the price predictions.
- **Model Evaluation**:
    - **Hyperparameter Optimization**:
        - Fine-tune the model using techniques such as **GridSearchCV** or **RandomizedSearchCV** to optimize the hyperparameters and improve model performance.
    - **Model Comparison**:
        - Compare the performance of different regression models and select the best-performing model based on evaluation metrics.
- **Deployment with Streamlit**:
    - **Streamlit Web App**:
        - Create a Streamlit app where users can input flight details such as **airline**, **departure city**, **destination city**, **class**, **stops**, and **date**.
    - **Prediction Display**:
        - Display the predicted flight price based on user inputs.
    - **Visualize Model Accuracy**:
        - Allow users to visualize model accuracy with **evaluation metrics** and **plots** showing the comparison of predicted vs actual prices.

# Technology Stack
* Python: This is the workhorse for data science due to its rich ecosystem of libraries, ease of use, and strong community support.
* Pandas: Essential for data manipulation and analysis, providing powerful data structures like DataFrames for cleaning, transforming, and exploring the dataset.
* NumPy: Fundamental for numerical computations in Python, especially when dealing with arrays and mathematical operations required for data preprocessing and model training.
* Scikit-learn (sklearn): A comprehensive machine learning library offering various regression algorithms (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor), tools for train-test splitting (train_test_split), model evaluation metrics (MAE, RMSE), and hyperparameter tuning (GridSearchCV, RandomizedSearchCV).
* Matplotlib and Seaborn: Libraries for creating insightful visualizations during Exploratory Data Analysis (EDA) to understand data distributions and relationships between features.
* Streamlit: Chosen for its simplicity and efficiency in creating interactive web applications with minimal coding. It allows you to quickly deploy your trained model and build a user interface for predictions and visualizations.

