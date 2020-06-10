# necessary imports 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  

# loading Hitters data
hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0)
hitters = hitters.dropna()

# create predictor matrix and response vector 
X = hitters.drop('Salary', axis=1)
X = pd.get_dummies(X, drop_first=True)
y = hitters.Salary

# testing/training split of data (25% for testing by default)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# convert to proper data type 
X_train = np.array(X_train)
X_test = np.array(X_test)

# fit to training data, only, and perform standardizing operations 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def objective(Bt, lmbda):
    diff = y_train - (X_train @ Bt)
    risk = (diff.T @ diff) / len(y_train)
    penalty = lmbda * (Bt.T @ Bt)
    return risk + penalty


def computegrad(B, lmbda):
    XTy = X_train.T @ y_train
    XTX = X_train.T @ X_train
    gradient = (-2 * XTy) + (2 * (XTX @ B)) + (2 * lmbda * B)
    return gradient 


def graddescent(n, T, lmbda):
    B_t = np.zeros(np.shape(X_train)[1])
    objective_values = []
    iter_num = 0 
    
    while iter_num < T:
        B_t = B_t - (n * computegrad(B_t, lmbda=lmbda))
        objective_values.append(objective(B_t, lmbda))
        iter_num += 1
    
    return objective_values, B_t


import matplotlib.pyplot as plt
objs = graddescent(0.05, 1000, -5)[0]
axis = np.array(range(len(objs)))
plt.scatter(axis, objs)


objs, Bt = graddescent(0.05, 1000, lmbda=0.05)
axis = np.array(range(len(objs)))
plt.scatter(axis, objs)


from sklearn import linear_model 

# sklearn Beta
sk_beta = linear_model.Ridge(alpha = 0.05).fit(X_train, y_train).coef_

# our Beta vs. sklearn
print(Bt)
print(sk_beta)


# objective values
print("Our iterate's obj. value: " + str(objective(Bt, 0.05)))
print("Sklearn obj. value: " + str(objective(sk_beta, 0.05)))


final_obj_values = []

for lr in [0.1, 0.01, 0.001]:
    my_Bt = graddescent(lr, 1000, 0.05)[1]
    final_obj_values.append(objective(my_Bt, 0.05))

# our best result vs. sklearn
print("Objective value for our best final iterate: " + str(min(final_obj_values)))
print("Objective value for sklearn final iterate: " + str(objective(sk_beta, 0.05)))


def e_graddescent(n, lmbda, e):
    B_t = np.zeros(np.shape(X_train)[1])
    objective_values = [] 
    grad = computegrad(B_t, lmbda)    
    
    while np.linalg.norm(grad) <= e:
        objective_values.appen(objective(B_t), lmbda)
        B_t = B_t - (n*computegrad(B_t, lmbda))
        
    return objective_values, B_t 


objs, Bf = e_graddescent(0.05, 0.05, 0.005)
plt.plot(objs)


# our Beta vs. sklearn
print(Bf)
print(sk_beta)


print("Our iterate's obj. value: " + str(objective(Bf, 0.05)))
print("Sklearn obj. value: " + str(objective(sk_beta, 0.05)))


# again trying different learning rates 
final_obj_values = []

for lr in [0.1, 0.01, 0.001]:
    my_Bt = e_graddescent(lr, 1000, 0.05)[1]
    final_obj_values.append(objective(my_Bt, 0.05))

print("Objective value for our best (final) Beta iterate: " + str(min(final_obj_values)))
print("Objective value for sklearn iterate: " + str(objective(sk_beta, 0.05)))


auto = pd.read_csv('../resources/Auto.csv', na_values='?')
auto = auto.dropna()


import statsmodels.api as sm 

# add column of 1s to obtain intercept 
predictors = sm.add_constant(auto.weight)
response = auto.mpg

# fit model and output results 
model = sm.OLS(response, predictors)
results = model.fit()
results.summary()


fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 1, ax=ax)
ax.set_title('MPG vs. Weight')
plt.show()


plt.scatter(x=results.fittedvalues, y=results.resid)
plt.xlabel('Fitted')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()


fig = pd.plotting.scatter_matrix(auto, figsize=(16,9))
plt.show()


auto.corr()


auto_X = sm.add_constant(auto.drop(['mpg', 'name'], axis = 1))
auto_y = auto.mpg
auto_model = sm.OLS(auto_y, auto_X).fit()
auto_model.summary()


plt.scatter(x=auto_results.fittedvalues, y=auto_results.resid)
plt.xlabel('Fitted')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()


import statsmodels.formula.api as smf
df = auto.drop(['name'], axis=1)
interaction_only_model = smf.ols(formula='mpg ~ weight : acceleration', data=df).fit()
interaction_only_model.summary()


# same as before, but we include the weight/acceleration columns, too
inclusive_model = smf.ols(formula='mpg ~ weight * acceleration', data=df).fit()
inclusive_model.summary()


# limit to continuous variables 
df = df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']]

log_df = np.log(df)
inv_df = df.apply(lambda x: 1/x)

pd.plotting.scatter_matrix(log_df, figsize=(8,6))
plt.show()


pd.plotting.scatter_matrix(inv_df, figsize=(8,6))
plt.show()

