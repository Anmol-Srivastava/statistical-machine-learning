import pandas as pd 
import numpy as np 

from sklearn.preprocessing import StandardScaler

PREFIX = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.'

# loading 
spam = pd.read_csv(PREFIX + 'data', sep=' ', header=None)
splits = pd.read_csv(PREFIX + 'traintest', header=None)


# train/test split 
merged = spam.merge(splits, left_index=True, right_index=True)
merged[57] = merged[57].apply(lambda x: 1 if x==1 else -1)

train = merged[merged['0_y'] == 0]
test = merged[merged['0_y'] == 1]

X_train = train.iloc[:,:-2]
X_test = test.iloc[:, :-2]

y_train = train.iloc[:, -2]
y_test = test.iloc[:, -2]

# standardize 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


""" FUNCTION DEFINITIONS, hyperparamsearch() IS IN LATER SECTION """

def obj(B, lmbda, X, y):
    risk = (1/len(y)) * np.sum(np.exp(-y*np.dot(X,B)))
    penalty = lmbda * np.linalg.norm(B)**2
    return risk + penalty 


def grad(B, lmbda, x, y):
    yx = y[:,np.newaxis]*x
    risk = (-1/len(y)) * np.sum(yx * np.exp(-np.dot(yx,B[:,np.newaxis])))
    penalty = 2*lmbda*B
    return risk + penalty 


def backtracking(B, lmbda, eta=1, alpha=0.5, gamma=0.8, T=100, X=X_train, y=y_train):
    grad_B = grad(B, lmbda, X, y)
    norm_grad_B = np.linalg.norm(grad_B)
    finished_bt = False 
    t = 0 
    
    while (not finished_bt) and (t < T):
        if obj(B-eta*grad_B, lmbda, X, y) < obj(B, lmbda, X, y) - alpha*eta*norm_grad_B**2:
            finished_bt = True 
        elif t == T:
            raise "Exceeding 100 iterations of backtracking line search."
        else:
            eta *= gamma
            t += 1
    
    return eta 


def myclassifier(epsilon, lmbda, X=X_train, y=y_train):
    w, v = np.linalg.eigh((1/len(y)) * np.dot(X.T, X))
    eta_init = 1 / (max(w) + lmbda) 

    beta = np.zeros(X.shape[1])
    theta = np.zeros(X.shape[1])   

    grad_theta = grad(theta, lmbda, X, y)
    grad_beta = grad(beta, lmbda, X, y)

    beta_vals = beta 
    t = 0 
    
    while np.linalg.norm(grad_beta) > epsilon and t < 1000: 
        eta = backtracking(beta, lmbda, eta=eta_init) 
        beta_new = theta - eta*grad_theta 
        theta = beta_new + (t/(t+3)) * (beta_new-beta)
        
        beta = beta_new 
        beta_vals = np.vstack((beta_vals, beta))
        grad_theta = grad(theta, lmbda, X, y)
        grad_beta = grad(beta, lmbda, X, y)
        t += 1 
    
    return beta_vals


""" TRAINING & TUNING """

# lambda=1, epsilon=0.005
Bs = myclassifier(0.005, 1)
B = Bs[-1]


# misclassification error
def misclassification_error(B, X, y):
    err = 0
    y = y if isinstance(y, np.ndarray) else y.to_numpy()
    
    for i in range(len(y)):
        if np.sign(np.dot(X[i], B)) != y[i]:
            err += 1

    return err/len(y) 

mc_err_train = 100 * misclassification_error(B, X_train, y_train)
mc_err_test = 100 * misclassification_error(B, X_test, y_test)

print('When lambda = 1,')
print('Misclassification Error, Training Data: %.2f%%' % mc_err_train)
print('Misclassification Error, Testing Data: %.2f%%' % mc_err_test)


from sklearn.model_selection import train_test_split

# grid search for lambda with 60/40 split for training/val
def hyperparamsearch():
    
    # limited range due to time-efficiency concerns 
    scale = np.logspace(-3, 0, 4)
    optimal_lambda = 0 
    optimal_err = 1
    optimal_B = None
    
    # merge and split data 
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    xtrn, xval, ytrn, yval = train_test_split(X, y, train_size=0.6)
    
    # search via validation data
    for lmbda in scale: 
        B = myclassifier(epsilon, lmbda, xtrn, ytrn)[-1]
        err = misclassification_error(B, xval, yval)
        
        if err < optimal_err:
            optimal_lambda = lmbda
            optimal_err = err 
            optimal_B = B
    
    # remaining metrics
    TP = 0
    TN = 0 
    
    for i in range(len(yval)):
        if np.sign(np.dot(xval[i], optimal_B)) == 1 and yval[i] == 1:
            TP += 1
        elif np.sign(np.dot(xval[i], optimal_B)) == -1 and yval[i] == -1:
            TN += 1
        else:
            pass 
    
    sens = 100*TP / len(yval[yval == 1]) 
    spec = 100*TN / len(yval[yval == -1])
    main = 100*optimal_err 
    
    print('Optimal value of lambda: %g. For this lambda,' % optimal_lambda)
    print('Misclassification error is: %.2f%%' % main)
    print('Sensitivity (in the validation set) is: %.2f%%' % sens)
    print('Specificity (in the validation set) is: %.2f%%' % spec)
    
hyperparamsearch()

