
# IPL-Score-Prediction-using-ML
The IPL score prediction project uses machine learning to forecast scores of IPL matches based on historical data. It involves data collection, preprocessing, and feature engineering, utilizing algorithms like linear regression and decision trees. Python is employed for model training and evaluation to enhance prediction accuracy.

# Why we using Deep Learning for Score Prediction?
We humans can't easily identify patterns from huge data, deep learning algorithms can do this efficiently. These algorithms learn from how players and teams have performed against each other in the past. Unlike simpler machine learning methods, deep learning can analyze many different factors at once making its score forecasts more reliable and closer to reality.

# Step 1 Installing Libraries
We are importing all necessary Python libraries such as NumPy, Pandas, Scikit-learn, Matplotlib, Keras and Seaborn required for data handling, visualization, preprocessing and building deep learning models.

# Step 2 Loading the Dataset
The dataset is uploaded in the repository. It contains data from 2008 to 2017 and contains features like venue, date, batting and bowling team, names of batsman and bowler, wickets and more. We will load the IPL cricket data from CSV files into pandas DataFrames to explore and prepare for modeling.

# Step 3 Dropping Unnecessary Features
Here, some columns are dropped from the data because they are not needed for prediction. For example: 'runs', 'wickets' etc., might be targets or irrelevant features for predicting the total score because the project is prematch analysis.

# Step 4 Separating Features (X) and Target (y)
•	X: Contains all the features (columns), excluding the 'total' column.
•	y: Contains the target variable, i.e., the 'total' score that we are trying to predict.

# Step 5 Label Encoding Categorical Features
•	LabelEncoder: Converts categorical values (like team names, venue names, etc.) into numerical values. This is necessary because machine learning models typically cannot handle non-numeric data directly.
•	fit_transform(): Fits the encoder to the data and transforms the categorical features into numerical values.

# Step 6 Splitting the Data into Training and Testing Sets
•	train_test_split(): Splits the data into training and testing sets. 70% of the data is used for training, and 30% is used for testing. This helps evaluate how well the model will generalize to unseen data.

# Step 7 Scaling the Data
•	MinMaxScaler: Scales the features so that they all fall within a specific range (usually 0 to 1). This is important for neural networks, as it ensures that features are on a similar scale and helps the model converge faster.
•	fit_transform(): This fits the scaler to the training data and transforms it.
•	transform(): This scales the test data using the scaler fitted on the training data.

# Step 8 Defining the Neural Network Model
Here, we define a neural network model:
•	Input Layer: Takes the scaled input features (number of features depends on the dataset).
•	Hidden Layers: These layers process the data. The first hidden layer has 512 neurons, and the second hidden layer has 216 neurons, both using the ReLU activation function (a common function for deep learning).
•	Output Layer: This layer outputs a single value, which represents the predicted score, using a linear activation function (since this is a regression problem).

# Step 9 Compiling the Model
•	Huber Loss: This is a loss function used for regression problems. It is less sensitive to outliers than Mean Squared Error (MSE).
•	Optimizer: We use the Adam optimizer because it works well in most cases and adjusts the learning rate during training.

# Step 10 Training the Model
•	fit(): This trains the model using the training data (X_train_scaled and y_train).
•	epochs=50: The model will go through the training data 50 times.
•	batch_size=64: The model processes 64 samples at a time during training.
•	validation_data: This is used to check the model’s performance on the test data (X_test_scaled, y_test) after each epoch.

# Step 11 Plotting Loss
•	This plots the training loss over the epochs to visually track how well the model is improving during training.
![Image](https://github.com/user-attachments/assets/051d1d8b-fca6-456c-a720-5926194e23b7)

# Step 12 Making Predictions
•	predict(): After training, the model is used to make predictions on the test data (X_test_scaled).

# Step 13 Evaluating the Model
•	mean_absolute_error(): This function computes the mean absolute error (MAE) between the true values (y_test) and predicted values (predictions). MAE tells us how much, on average, the predictions deviate from the true values.
![Image](https://github.com/user-attachments/assets/61e80bf0-5d9a-415d-9d3b-f753f1a4ee0d)

# Step 14 Interactive Widgets for Prediction
This section sets up interactive widgets for users to select values for the model's input (venue, teams, etc.). This allows you to make predictions based on user input.

# Step 15 The Prediction Function
•	predict_score(b): This function is triggered when the "Predict Score" button is pressed. It:
o	Decodes the user-selected categorical inputs.
o	Scales the inputs.
o	Makes a prediction using the model and prints the predicted score.
![Image](https://github.com/user-attachments/assets/7fd3f071-a57f-41f5-a33c-c6ae1a220057)
