import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler
import sklearn.metrics 

""" DATA PROCESSING """

# access data, concatenate to split later 
df_train = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.train')
df_test = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test')
df = pd.concat([df_train, df_test])

# drop extra index 
df = df.drop('row.names', axis=1)

# select '1' as positives and '2', '3' as negatives, drop rest, apply +/- 1 labels
df = df[df.y.between(1, 3, inclusive=True)] 
df.y = df.y.apply(lambda x: 1 if x==1 else -1)

# splitting data (70/15/15 ratio)
n = len(df)

# note: data is ordered in increasing pattern by response
# (y=1, 2, ... 11, 1, 2, ... 11), hence split automatically provides 
# desired similarity in positive/negative-proportions for each set 
train, validate, test = np.split(df, [int(0.7*n), int(0.85*n)])

y_train = train.y.to_numpy()
y_val = validate.y.to_numpy()
y_test = test.y.to_numpy()

X_train = train.drop('y', axis=1).to_numpy()
X_val = validate.drop('y', axis=1).to_numpy()
X_test = test.drop('y', axis=1).to_numpy()

# standardize based off of training set 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


""" FUNCTIONS """

def huber_loss(y, t, h=0.5):
    lhh = None
    if y*t > 1+h:
        lhh = 0
    elif np.abs(1-(y*t)) <= h:
        lhh = ((1+h-(y*t))**2) / (4*h)
    elif y*t < 1-h:
        lhh = 1-(y*t)
    else: 
        print('Parameter failure in Huber loss function.')
    return lhh 


def grad_huber_loss(y, X, B, h=0.5):
    # expect col. vector X
    t = np.dot(X,B)
    grad_lhh = None
    if y*t > 1+h:
        grad_lhh = np.zeros(X.shape)
    elif np.abs(1-(y*t)) <= h:
        grad_lhh = (1/(2*h)) * (1+h-(y*t)) * (-y*X) 
    elif y*t < 1-h:
        grad_lhh = -y*X
    else: 
        print('Parameter failure in Huber loss function.')
    return grad_lhh 


def obj(B, rho, lmbda, X, y):
    penalty = lmbda * np.linalg.norm(B)**2
    pos_term = 0
    neg_term = 0 
    
    for i in np.where(y == 1)[0]:
        pos_term += huber_loss(y[i], np.dot(X[i], B))
    
    for i in np.where(y == -1)[0]:
        neg_term += huber_loss(y[i], np.dot(X[i], B)) 
    
    pos_term *= rho/len(y)
    neg_term *= (1-rho)/len(y)
    return penalty + pos_term + neg_term 


def grad(B, rho, lmbda, X, y):
    penalty = 2*lmbda*B
    pos_term = np.zeros(X.shape[1])
    neg_term = np.zeros(X.shape[1])
    
    for i in np.where(y == 1)[0]: 
        pos_term += grad_huber_loss(y[i], X[i], B)
    
    for i in np.where(y == -1)[0]:
        neg_term += grad_huber_loss(y[i], X[i], B)
    
    pos_term *= rho/len(y)
    neg_term *= (1-rho)/len(y)
    return penalty + pos_term + neg_term 


def backtracking(B, rho, lmbda, X, y, eta=1, alpha=0.5, gamma=0.8, max_iter=100):
    grad_B = grad(B, rho, lmbda, X, y)
    norm_grad_B = np.linalg.norm(grad_B)
    found_eta = 0
    num_iters = 0
    
    while found_eta == 0 and num_iters < max_iter:
        if obj(B-eta*grad_B, rho, lmbda, X, y) < obj(B, rho, lmbda, X, y)-alpha*eta*norm_grad_B**2:
            found_eta = 1
        elif num_iters == max_iter:
            raise('Max. iteration of BLS reached.')
        else:
            eta *= gamma
            num_iters += 1 
            
    return eta


def mylinearsvm(rho, lmbda, X, y, eps=5e-3):
    # init eta using L constant as described in class 
    w, v = np.linalg.eigh((1/len(y)) * np.dot(X.T, X))
    
    beta = np.zeros(X.shape[1])
    theta = np.zeros(X.shape[1])    
    eta = 1 / (max(w) + lmbda) 
    
    grad_theta = grad(theta, rho, lmbda, X, y)
    
    grad_beta = grad(beta, rho, lmbda, X, y)
    
    beta_vals = beta 
    t = 0 
    
    while np.linalg.norm(grad_beta) > eps:
        # hw3 soln used theta instead of beta as arg here -- unsure if error
        eta = backtracking(beta, rho, lmbda, X, y, eta=eta) 
        beta_new = theta - eta*grad_theta 
        theta = beta_new + (t/(t+3)) * (beta_new-beta)
        
        beta_vals = np.vstack((beta_vals, beta))
        grad_theta = grad(theta, rho, lmbda, X, y)
        grad_beta = grad(beta, rho, lmbda, X, y)
        beta = beta_new
        t += 1 
    
    return beta_vals 


""" TRAINING & TESTING """

# rho=1, lambda=1
final_beta = mylinearsvm(1, 1, X_train, y_train)[-1]

def performance_metrics(B, X, y, output=True):
    P = len(np.flatnonzero(y==1))
    N = len(np.flatnonzero(y==-1))
    TN = 0
    FN = 0
    TP = 0
    FP = 0
        
    for i in range(len(y)):
        if int(np.sign(np.dot(X[i], final_beta))) == 1:
            TP = TP+1 if y[i]==1 else TP
            FP = FP+1 if y[i]==-1 else FP
        else:
            TN = TN+1 if y[i]==-1 else TN
            FN = FN+1 if y[i]==1 else FN 
    
    misclassed = 100*(FP+FN)/len(y)
    sens = 100*TP/P
    spec = 100*TN/N
    
    if output:
        print('General misclassification error is: %.2f%%.' % misclassed)
        print('Achieved sensitivity is: %.2f%%.' % sens)
        print('Achieved specificity is: %.2f%%.' % spec)
            
    return (misclassed, sens, spec)


# feedback on rho=1, lambda=1
metrics = performance_metrics(final_beta, X_train, y_train)


# lambda=1, rho=0.1 to 1.0
rhos = [x/10 for x in list(range(1,11))]
lmbda = 1 

misclass_errs = []
sensitivities = []
specificities = [] 

for rho in rhos:
    final_beta = mylinearsvm(rho, lmbda, X_train, y_train)[-1]
    metrics = performance_metrics(final_beta, X_train, y_train, output=False)
    misclass_errs.append(metrics[0]/100)
    sensitivities.append(metrics[1]/100)
    specificities.append(metrics[2]/100)

plt.plot(rhos, misclass_errs, label="Misclassification Error")
plt.plot(rhos, sensitivities, label="Sensitivity")
plt.plot(rhos, specificities, label="Specificity")
plt.title('Performance Metrics vs. Imbalance Parameter')
plt.xlabel('Imbalance Parameter')
plt.ylabel('Metric Value')
plt.legend()
plt.show()


# varying both lambda and rho
lambdas = rhos 

def predict(B, X=X_val, y=y_val):
    predictions=np.array([])
    
    for i in range(len(y)):
        predictions = np.append(predictions, int(np.sign(np.dot(X[i], B))))
        
    return predictions 


# makeshift grid search 
optimal_roc_auc = 0
optimal_rho = -1
optimal_lambda = -1 
optimal_B = -1

for l in lambdas:
    for r in rhos: 
        Bf = mylinearsvm(r, l, X_val, y_val)[-1]
        y_preds = predict(Bf)
        score = sklearn.metrics.roc_auc_score(y_val, y_preds)
        
        if score > optimal_roc_auc:
            optimal_roc_auc = score
            optimal_rho = r
            optimal_lambda = l
            optimal_B = Bf
            

print('For the validation set:\n\nThe optimal value of lambda is: %.1f.' % optimal_lambda)
print('The optimal value of rho is: %.1f.' % optimal_rho)
print('The corresponding, optimized area-under-the-ROC-curve is: %.2f.' % optimal_roc_auc)

metrics = performance_metrics(optimal_B, X_test, y_test, output=False)
print('\nFor the testing set (using above optimal parameters):\n')
print('The achieved sensitivity is: %.2f%%' % metrics[1])
print('The achieved specificity is: %.2f%%' % metrics[2])




