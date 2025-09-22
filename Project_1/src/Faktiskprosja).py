
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures # use the fit_transform method of the created object!
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

n = 100
x = np.linspace(-1,1 , n)
y = 1/(1+25*x**2)
yrand = y +np.random.normal(0, 1)

def polynomial_features(x, p, intercept =True):
    k = 1
    if intercept == True:
        k = 0

    n = len(x)
    X = np.zeros((n, p +1-k))


    for i in range(k,p+1):
        def f(x):
            return x**i
        X[:,i-k] = f(x)
    return X

X = polynomial_features(x, 5, intercept=True)



Xtest = PolynomialFeatures(degree=5,include_bias=False)
testpoly=Xtest.fit_transform(x.reshape(-1, 1))

print(np.shape(Xtest))
print(np.shape(testpoly))


def Ridge_parameters(X, y,lam):
    # Assumes X is scaled and has no intercept column
    Lambda = lam
    A= X.T @ X
    I = np.eye(*A.shape)
    return np.linalg.inv(A+ Lambda*I) @ X.T @ y





def testRidge(x,p,lam,state):
    '''
    Function to test the effectiveness of the model with OLS
    '''
    X = polynomial_features(x, p, state)
    var = 0
    if state== True:
        var = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    x_train = X_train[:, var] # These are used for plotting
    x_test = X_test[:, var] # These are used for plotting

    '''
    Scaling data so that we can take a ridge regression afterwards
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    y_offset = np.mean(y_train)

    beta = Ridge_parameters(X_train_s,y_train,lam)

    '''
    X_train is the training data that is taken out for X
    '''

    mod = LinearRegression().fit(X_train, y_train)
    X_predictTrain = mod.predict(X_train)
    X_predictTest = mod.predict(X_test)


    predicmeansquareTrain = mean_squared_error(y_train, X_predictTrain)
    predicmeansquareTest = mean_squared_error(y_test, X_predictTest)

    # print(f'The training mst is {predicmeansquareTrain}: The test mst is {predicmeansquareTest}')
    return predicmeansquareTrain, predicmeansquareTest, X_predictTest, X_predictTest, X_train, X_test, beta, x_train, x_test, X_train_s,X_test_s, y_offset
    
def msetestdeg(deg):
    '''
    Making a loop of degrees of polynomials to take OLS on the model.
    '''
    max_deg = deg
    mseTrain = np.zeros(max_deg)
    mseTest = np.zeros(max_deg)

    degs = np.linspace(2,max_deg,max_deg)

    for i in range(2, max_deg+1):
        mseTrain[i-2], mseTest[i-2], _,_,_,_,_= testRidge(i)

    '''
    Plotting the effectiveness for each polynomial
    '''

    plt.plot(degs,mseTest,label= "Test")
    plt.plot(degs, mseTrain, label = "Train")
    plt.legend()
    plt.show()



p = 5
plist = np.linspace(1,p,p)
mseTest = np.zeros_like(plist)
mseTrain= mseTest



plt.plot(plist,mseTest)
plt.plot(plist, mseTrain)
plt.show()