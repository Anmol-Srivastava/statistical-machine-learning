import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegressionCV

""" Exercise 1 """

# load data
spam = pd.read_csv('../resources/spam.data', sep=' ', header=None)

# change output labels to -1/+1
y = spam.iloc[:,-1].apply(lambda x: -1 if x==0 else 1)

# standardize features (by default, done independently)
spam_arr = np.array(spam.iloc[:,:-1])
scaler = StandardScaler()
X = scaler.fit_transform(spam_arr)


def obj(B, lmbda, X=X, y=y):
    risk = (1/len(y)) * np.log(1 + np.exp(-np.dot(y,np.dot(X,B))))
    penalty = lmbda * np.linalg.norm(B)**2
    return risk + penalty 


def computegrad(B, lmbda=0, X=X, y=y):
    exp_term = np.exp(-np.dot(y, np.dot(X,B))) 
    risk = (-1/len(y)) * np.dot(np.dot(y,X), exp_term) / (1 + exp_term)
    penalty = 2 * lmbda * B
    return risk + penalty


def backtracking(B, eta, alpha=0.5, gamma=0.8, lmbda=0, X=X, y=y):
    grad_B = computegrad(B, X=X, y=y)
    norm_grad_B = np.linalg.norm(grad_B)    
    
    while not obj(B-eta*grad_B, lmbda, X=X, y=y) <= obj(B,lmbda, X=X, y=y) - alpha*eta*norm_grad_B**2:
        eta *= gamma
          
    return eta


def graddescent(B_init, eta_init, epsilon, lmbda=0, max_iter=1000, X=X, y=y):   
    eta = eta_init 
    B = B_init 
    grad_B = computegrad(B, lmbda=lmbda, X=X, y=y)
    B_vals = [B]
    iter_num = 0 
    
    while np.linalg.norm(grad_B) > epsilon:
        eta = backtracking(B, eta, lmbda=lmbda, X=X, y=y)
        B = B - eta*grad_B
        B_vals.append(B)    
        grad_B = computegrad(B, lmbda=lmbda, X=X, y=y)
        iter_num += 1
    
    return np.array(B_vals) 


def fastgradalgo(eta_init, epsilon, lmbda=0, max_iter=1000, X=X, y=y):
    theta = np.zeros(X.shape[1])
    B = np.zeros(X.shape[1])
    grad_B = computegrad(B, lmbda=lmbda, X=X, y=y)
    B_vals = [B]
    t = 0 
    eta = eta_init 
    
    while np.linalg.norm(grad_B) > epsilon:
        B_t = B
        eta = backtracking(B_t, eta, lmbda=lmbda, X=X, y=y)
        B = theta - eta*computegrad(theta, lmbda=lmbda, X=X, y=y)
        theta = B + (t/(t+3)) * (B-B_t)
        
        B_vals.append(B)
        grad_B = computegrad(B, lmbda=lmbda, X=X, y=y)
        t += 1
      
    return np.array(B_vals)


# regularization coefficient 
lmbda = 0.5

# n0 = 1/L-constant as in class 
w, v = np.linalg.eigh((1/len(y)) * np.dot(X.T, X))
n0 = 1 / (max(w) + lmbda)  

# target accuracy
epsilon = 5e-3

# for standard gradient descent, B0 is the zero vector
B = np.zeros(X.shape[1])

# access B iterates for both algorithms 
gd_betas = graddescent(B, n0, epsilon, lmbda=lmbda)
fg_betas = fastgradalgo(n0, epsilon, lmbda=lmbda)


def objective_plots(gd_betas, fg_betas, lmbda, X=X, y=y):
    n1, _ = gd_betas.shape
    n2, _ = fg_betas.shape 
    f1s = np.zeros(n1)
    f2s = np.zeros(n2)
    
    for i in range(n1):
        f1s[i] = obj(gd_betas[i], lmbda, X=X, y=y)
    
    for i in range(n2):
        f2s[i] = obj(fg_betas[i], lmbda, X=X, y=y)
        
    plt.plot(f1s, label='graddescent')
    plt.plot(f2s, label='fastgradalgo')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('F(B) Over Time')
    plt.legend()

    
objective_plots(gd_betas, fg_betas, 0.5)


# final iterates and obj values for fastgradalgo 
B_T = fg_betas[-1]
obj_Bt = "{:.2E}".format(obj(B_T, lmbda))

# sklearn model
C = [1/(2*len(y)*lmbda)]

classifier = LogisticRegressionCV(Cs=C, fit_intercept=False)
classifier.fit(X,y)

B_S = classifier.coef_.reshape((57,))
obj_Bs = "{:.2E}".format(obj(B_S, lmbda))

print("Our final iterate, B_T:\n")
print(B_T)
print("\nEstimate from sklearn:\n")
print(B_S)


print("Objective value for our final iterate, B_T:\n")
print(obj_Bt)
print("\nObjective value for sklearn estimate:\n")
print(obj_Bs)


# use sklearn's calculation of optimal C, and correspondence to lambda
optimal_lmbda = 1 / (2*len(y)*classifier.C_[0]) 

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

new_gd_betas = graddescent(B, n0, epsilon, lmbda=optimal_lmbda, X=X_train, y=y_train)
new_fg_betas = fastgradalgo(n0, epsilon, lmbda=optimal_lmbda, X=X_train, y=y_train)

objective_plots(new_gd_betas, new_fg_betas, optimal_lmbda, X=X_train, y=y_train)


def error_plots(train_flag=True):
    x = X_train if train_flag else X_test
    y = y_train if train_flag else y_test
    
    subtitle = "Training Data" if train_flag else "Testing Data"
    title = "Misclassification Error vs. Iterations, %s" % subtitle
    
    gd_errs = []
    fg_errs = []
    
    # missing function to transform B,X into a 0/1 y 
    for item in new_gd_betas:
        pass 
    
    for item in new_fg_betas:
        pass 
    
    return gd_errs, fg_errs


# error_plots(train_flag=True)


# error_plots(train_flag=False)


optimal_lmbda

