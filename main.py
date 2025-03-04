# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Create X (After completing the exercise, you can return to modify this line!)
features = ['LotArea', 
            'YearBuilt', 
            '1stFlrSF', 
            '2ndFlrSF', 
            'FullBath', 
            'BedroomAbvGr', 
            'TotRmsAbvGrd']

# Select columns corresponding to features, and preview the data
X = home_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# n_estimators Define function to calculate MAE with a given n_estimators for 
# RandomForestRegressor

def mae_calc(n_estimators, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators = n_estimators, random_state = 1)
    model.fit(train_X, train_y)
    pred_y = model.predict(val_X)
    mae = mean_absolute_error(pred_y, val_y)
    return(mae)

# Check if results are the same as validation MAE from before
mae_calc(n_estimators = 100, 
         train_X = train_X, 
         val_X = val_X, 
         train_y = train_y, 
         val_y = val_y)

# Find n_estimators with the lowest MAE
best_mae = 0
best_n_estimators = 0
for n_estimators in [50, 100, 150, 200, 250, 300]:
    my_mae = mae_calc(n_estimators, train_X, val_X, train_y, val_y)
    if best_mae == 0:
        best_mae = my_mae
        best_n_estimators = n_estimators
    elif best_mae > my_mae:
        best_mae = my_mae
        best_n_estimators = n_estimators
print("Best MAE: %d\nBest Number of Trees: %d" %(best_mae, best_n_estimators))
# 100 (default) stays the best number of trees
