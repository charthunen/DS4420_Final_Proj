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

# store traditional metrics
rmse_trad = rmse
mae_trad  = mae
r2_trad   = r2

# load enriched data
data_enrich    = pd.read_csv('data/movies_clean_enriched.csv')
enriched_cols  = feature_cols + ['google_trends_interest']

Phi_e       = data_enrich[enriched_cols].to_numpy()
y_e         = data_enrich['log_revenue'].to_numpy()

scaler_e    = MinMaxScaler()
Phi_e_scale = scaler_e.fit_transform(Phi_e)

n_e         = Phi_e_scale.shape[0]
Phi_e_scale = np.column_stack([np.ones(n_e), Phi_e_scale])

[Phi_e_train, Phi_e_test, y_e_train, y_e_test] = train_test_split(
    Phi_e_scale, y_e, test_size=0.2, random_state=42)

# train enriched MLP
X_e  = Phi_e_train
p_e  = X_e.shape[1]

W1_e = np.random.randn(p_e, q) * np.sqrt(2/p_e)
W2_e = np.random.randn(q, 1)   * np.sqrt(2/q)

def f_e(x):
    h = relu(W1_e.T.dot(x))
    return W2_e.T.dot(h)

errors_e = []
n_e_tr   = X_e.shape[0]

for epoch in range(epochs):
    dW2_e = 0
    for i in range(n_e_tr):
        x = np.reshape(X_e[i], (p_e, 1))
        h = relu(W1_e.T.dot(x))
        dW2_e += (1/n_e_tr) * (f_e(x) - y_e_train[i]) * h
    W2_e = W2_e - eta * dW2_e

    dW1_e = 0
    for i in range(n_e_tr):
        x = np.reshape(X_e[i], (p_e, 1))
        h = relu(W1_e.T.dot(x))
        mat1 = np.heaviside(h, 0)
        dW1_e += (1/n_e_tr) * (f_e(x) - y_e_train[i]) * np.kron(x, (W2_e * mat1).T)
    W1_e = W1_e - eta * dW1_e

    loss_e = 0
    for i in range(n_e_tr):
        x = np.reshape(X_e[i], (p_e, 1))
        loss_e += (f_e(x)[0, 0] - y_e_train[i]) ** 2
    errors_e.append(loss_e / n_e_tr)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, MSE: {errors_e[-1]:.4f}")

# enriched test metrics
preds_e = np.array([f_e(np.reshape(Phi_e_test[i], (p_e, 1)))[0, 0]
                    for i in range(Phi_e_test.shape[0])])

rmse_enrich = np.sqrt(np.mean((preds_e - y_e_test) ** 2))
mae_enrich  = np.mean(np.abs(preds_e - y_e_test))
r2_enrich   = 1 - np.sum((y_e_test - preds_e) ** 2) / np.sum((y_e_test - np.mean(y_e_test)) ** 2)

# comparison
print("\n--- traditional vs enriched ---")
print(f"RMSE: {rmse_trad:.4f} -> {rmse_enrich:.4f}")
print(f"MAE:  {mae_trad:.4f}  -> {mae_enrich:.4f}")
print(f"R2:   {r2_trad:.4f}   -> {r2_enrich:.4f}")