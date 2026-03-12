import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

np.random.seed(42)

data = pd.read_csv('data/movies_clean.csv')
data.head()

feature_cols = [
    'log_budget','runtime','popularity','vote_average','vote_count',
    'num_genres','num_cast','is_major_studio','num_companies',
    'is_summer','is_holiday','is_franchise','is_english',
    'genre_action','genre_adventure','genre_animation','genre_comedy',
    'genre_crime','genre_drama','genre_family','genre_fantasy',
    'genre_horror','genre_romance','genre_science_fiction','genre_thriller'
]

Phi = data[feature_cols].to_numpy()
y = data['log_revenue'].to_numpy()

# scale 

scaler = MinMaxScaler()
Phi_scale = scaler.fit_transform(Phi)

# add bias column
n = Phi_scale.shape[0]
Phi_scale = np.column_stack([np.ones(n), Phi_scale])

# train/test split

[Phi_train, Phi_test, y_train, y_test] = train_test_split(Phi_scale, y, test_size=0.2)

# MLP setup 

X = Phi_train
y_tr = y_train

p = X.shape[1] 
q = 32           
eta = 0.001      
epochs = 500

# initialize weights 
W1 = np.random.randn(p, q) * np.sqrt(2/p)
W2 = np.random.randn(q, 1) * np.sqrt(2/q)

# relu activation
def relu(z):
    return np.maximum(0, z)

# prediction function
def f(x):
    h = relu(W1.T.dot(x))
    return W2.T.dot(h)

# training with gradient descent 
errors = []
n = X.shape[0]

for epoch in range(epochs):
    # update W2
    dW2 = 0
    for i in range(n):
        x = np.reshape(X[i], (p, 1))
        h = relu(W1.T.dot(x))
        dW2 += (1/n) * (f(x) - y_tr[i]) * h

    W2 = W2 - eta * dW2

    # update W1
    dW1 = 0
    for i in range(n):
        x = np.reshape(X[i], (p, 1))
        h = relu(W1.T.dot(x))
        mat1 = np.heaviside(h, 0)  # relu derivative
        dW1 += (1/n) * (f(x) - y_tr[i]) * np.kron(x, (W2 * mat1).T)

    W1 = W1 - eta * dW1

    # calculate MSE
    loss = 0
    for i in range(n):
        x = np.reshape(X[i], (p, 1))
        pred = f(x)[0, 0]
        loss += (pred - y_tr[i]) ** 2
    errors.append(loss / n)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, MSE: {errors[-1]:.4f}")

# plot convergence

plt.plot(range(epochs), errors)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MLP Training Loss')
plt.show();

# test accuracy 

test_predictions = []
for i in range(Phi_test.shape[0]):
    x = np.reshape(Phi_test[i], (p, 1))
    pred = f(x)[0, 0]
    test_predictions.append(pred)

test_predictions = np.array(test_predictions)

# RMSE
rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
print(f"Test RMSE: {rmse:.4f}")

# MAE
mae = np.mean(np.abs(test_predictions - y_test))
print(f"Test MAE: {mae:.4f}")

# R-squared
ss_res = np.sum((y_test - test_predictions) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"Test R-squared: {r2:.4f}")

# still need to:
# - train second model with digital signal features (google trends + twitter)
# - compare metrics between traditional vs enriched
# - save results 