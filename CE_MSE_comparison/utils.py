import numpy as np
import pandas as pd
import time
from sklearn import datasets
from sklearn.utils import shuffle 
from scipy import optimize
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z)*(1-sigmoid(z))
def costFunction(theta, X, y):
    '''
    We define here the cross entropy loss function
    takes X, y and theta as parameters. the values 
    of theta define the model it self.
    Their dimensions are as follows:
    X : (m,2), y: (m,1), theta: (1,2)
    
    returns the loss and the gradient of the loss
    '''
    #constants
    m = len(y)  
    
    #initialize loss J and its gradient
    J = 0
    grad = np.zeros(theta.shape)
    
    # cost function using vectorization
    z = np.matmul(theta,X.T)
    pred = sigmoid(z)
    
    # for visualization purposes we do not want the pred
    # to take 1 or 0 values so that the log(pre) and 
    # log(1-pred) do not taake -inf values. 
    pred[pred > 0.999] = pred[pred>0.999] - 1e-10
    pred[pred < 0.001] = pred[pred<0.001] + 1e-10
    
    # cross entropy loss
    J = (-1/m)*((np.dot(y,np.log(pred)))+ np.dot(1-y,np.log(1-pred))) 
    
    # the gradient of the J with respect to theta
    grad = (1/m) * np.matmul((pred-y).T,X)
    
    
    return J, grad

def costFunction_MSE(theta, X,y):
    '''
    We define here the mean squared error loss function.
    it takes X, y and theta as parameters. the values 
    of theta define the model it self.
    Their dimensions are as follows:
    X : (m,2), y: (m,1), theta: (1,2)
    
    returns the loss and the gradient of the loss
    '''
    #constants
    m = len(y)
    
    #initialize J and its gradient
    J = 0
    grad = np.zeros(theta.shape)
    
    # cost function using vectorization
    
    z = np.matmul(theta,X.T)
    pred = sigmoid(z)
    
    J = (1/(2*m))* np.dot(y-pred, y-pred)
    
    # gradient of J with respect to theta has extra
    # step of computing h'
    hp = sigmoid_grad(z)
    grad = (-1/m) * np.matmul((hp*(y-pred)).T,X)
    
    return J, grad


def gradientDescent(costFunc, X, y, theta, alpha, num_iter):
    '''
    The algorithm takes X, y and theta as well as
    alpha (the learning rate or step size) and the 
    number of iterations
    It returns the optimal theta that minimizes the loss
    function as well as a record of loss values and thetas
    per iteration. 
    '''
    m = len(y)
    J_history = []
    theta_hist = []
    for itern in range(num_iter):
        J,grad = costFunc(theta, X,y)
        theta = theta - alpha*grad
        J_history.append(J)
        theta_hist.append(theta)
        if itern > 100:
        # stop if improvement in theta is at machine epsilon scale
          theta_step = theta_hist[-1] - theta_hist[-2]
      
          if (abs(theta_step[0]) < 1/(2**50)) & (abs(theta_step[1]) < 1/(2**50)):
            print('Early stopping : theta step close to machine epsilon after '+str(len(theta_hist))+' iterations')
            return theta, J_history, theta_hist
        
    return theta, J_history, theta_hist

def gradientDescent_MSE(X, y, theta, alpha, num_iter):
    '''
    The algorithm takes X, y and theta as well as
    alpha (the learning rate or step size) and the 
    number of iterations
    It returns the optimal theta that minimizes the loss
    function as well as a record of loss values and thetas
    per iteration. 
    '''
    m = len(y)
    J_history = np.zeros(num_iter)
    theta_hist = []
    for itern in range(num_iter):
        J,grad = costFunction_MSE(theta, X,y)
        theta = theta - alpha*grad
        J_history[itern] = J
        theta_hist.append(theta)
        
    return theta, J_history, theta_hist

# stochastic gradient descent.

def SGD(costFunc, X,y, theta, alpha, epochs):
    m=len(X)
    J_hist = []
    theta_hist = []
    for epoch in range(epochs): 
        for e in range(m):
            Xx= X[e,].reshape((1,2)) # reshape X for each of vectorized compute
            yy = y[e].reshape((1,))
            
            #compute cost and gradient
            J, grad = costFunc(theta, Xx, yy)
            theta = theta - alpha * grad
            #keep record
            J_hist.append(J)
            theta_hist.append(theta)
        # check theta is improving
        theta_step = theta_hist[-2] - theta_hist[-1]
        
        # stop if improvement in theta is at machine epsilon scale
        if (abs(theta_step[0]) < 1/(2**50)) & (abs(theta_step[1]) < 1/(2**50)):
            print('Early stopping : theta step close to machine epsilon after '+str(len(theta_hist))+' iterations')
            return theta, J_hist, theta_hist
        
        if epoch % 10000 == 0:
            print('epoch : {},  theta : {},  loss : {:0.8f} '.format(epoch, theta, J))
                     
    return theta, J_hist, theta_hist 

#Mini-Batch gradient descent
def make_mini_batches(X,y,batch_size):
    #prepares mini-batches for gradient descent
    m = len(X)
    num_batches = m // batch_size
    mini_batches = []
    for k in range(num_batches):
        X_mini_batch = X[k*batch_size:(k+1)*batch_size,:]
        y_mini_batch = y[k*batch_size:(k+1)*batch_size]
        mini_batches.append((X_mini_batch,y_mini_batch))
    if m % batch_size != 0:
        X_mini_batch = X[num_batches * batch_size:,]
        y_mini_batch = y[num_batches * batch_size:]
        mini_batches.append((X_mini_batch, y_mini_batch))
    return mini_batches

def mini_bGD(costFunc, X, y, theta, batch_size, alpha, epochs):
    '''
    Runs gradient descent algorithm using mini-batches instead of 
    whole dataset.    
    '''
    J_hist = []
    theta_hist = []
    # make mini-batches
    mini_batches = make_mini_batches(X,y, batch_size)    
    for epoch in range(epochs):
        for mini_b in mini_batches:
            X,y = mini_b
            J, grad = costFunc(theta, X, y)
            theta = theta - alpha * grad
            J_hist.append(J)
            theta_hist.append(theta)
        # check theta is improving
        theta_step = theta_hist[-2] - theta_hist[-1]
        
        # stop if improvement in theta is at machine epsilon scale
        if (abs(theta_step[0]) < 1/(2**50)) & (abs(theta_step[1]) < 1/(2**50)):
            print('Early stopping : theta step close to machine epsilon after '+str(len(theta_hist))+' iterations')
            return theta, J_hist, theta_hist
        
        
        if epoch % 10000 == 0:
            print('epoch : {},  theta : {},  loss : {:0.8f} '.format(epoch, theta, J))
    return theta, J_hist, theta_hist 

def results(X_test,y_test, theta):
    # compute sigmoid values/pred probabilities using given(optimal) theta
    pred_test = sigmoid(np.dot(theta, X_test.T))
    # compute predicted class/label
    pred_y_test = np.zeros(y_test.shape)
    pred_y_test[pred_test >= 0.5] = 1
    # compute confusion matrix values
    true_values = np.sum(y_test ==  pred_y_test)
    FN = np.sum((y_test == 1) & (pred_y_test == 0))
    FP = np.sum((y_test == 0) & (pred_y_test == 1))
    TN =  np.sum((y_test == 0) & (pred_y_test == 0))
    TP = np.sum((y_test == 1) & (pred_y_test == 1))
    # report results
    print('False negatives ', FN)
    print('false positive ', FP)
    print('true negatives ', TN)
    print('true positives ', TP)
    print('total true values ', TN+TP)
    print('\n')
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print('precision {0:1.4f}'.format(precision))
    print('recall {0:1.4f}'.format(recall))
    print('accuracy % 2.4f ' % (true_values/len(y_test)))
    print('F1 score % 1.4f ' % (2*precision*recall/(precision+recall)))
    
    
 # slices of cross entropy loss
# this function creates data needed for plotting profiles/slices

def get_profile(costFunc,X_train_o, y_train, th1, th2, ths1, ths2, axis):
    '''
     Parameters:
     costFunc: the name of the cost function
     th1 : values of theta_0 used to plot the profile
     th2 : values of theta_1 used to plot the profile
     ths1 & ths2 : values of theta_0 and theta_1 respectively
                   generated by the gradient descent alogorithm
                   along the path of convergence to optimal theta
     axis: detemines the direction of the slice. 
           expected values are 'theta_0' or 'theta_1'
     Returns:
      J_t1: values of the cost function plotting the profile
      J_tt1 values of the cost function plotting the points 
            generated by the gradient descent algorithm.
    '''
    if axis == 'theta_0':
        J_t1 = np.zeros(th1.shape)
        for i in range(len(th1)):
            th = np.array([th1[i],th2])
            J_t1[i], _ = costFunc(th,X_train_o,y_train)

        J_tt1 = np.zeros(len(ths1))
        for i, t1 in enumerate(ths1):
            th = np.array([t1,th2])
            J_tt1[i], _ =costFunc(th,X_train_o,y_train)
        return J_t1, J_tt1
    
    elif axis == 'theta_1':
        J_t2 = np.zeros(th2.shape)
        for i in range(len(th2)):
            th = np.array([th1,th2[i]])
            J_t2[i], _ = costFunc(th, X_train_o,y_train)

        J_tt2 = np.zeros(len(ths2))
        for i, t2 in enumerate(ths2):
            th = np.array([th1,t2])
            J_tt2[i], _ =costFunc(th,X_train_o,y_train)
        return J_t2, J_tt2
    
    
    
# preparing plots of profiles
def plot_profiles(th1_1, th2_2, ths1, ths2, J_t1, J_tt1, J_t2, J_tt2, th2_1, th1_2,loss_type = 'CE'):
    plt.subplots(1,2,figsize =(12,4))

    plt.subplot(121)
    plt.plot(th1_1, J_t1)
    plt.plot(ths1, J_tt1, 'r*')
    plt.title('profile along theta_0 where theta_1 is fixed at '+str(round(th2_1,5)))
    plt.xlabel('theta_0')
    plt.ylabel('Loss')


    plt.subplot(122)
    plt.plot(th2_2, J_t2)
    plt.plot(ths2, J_tt2,'*r')
    plt.title('profile along theta_1 where theta_0 is fixed at '+str(round(th1_2,5)))
    plt.xlabel('theta_1')
    plt.ylabel('Loss')

    plt.suptitle('Profile of the '+str(loss_type)+' loss function \n in red are the theta values of few iterations as projected on the profile')
    plt.tight_layout()
    plt.show()
    
    
def local_min_contour_plot(costFunc,X_train_o, y_train, theta_0_lcm, iterations, xmin, xmax, ymin, ymax, gmin, \
                           plot_gradient = True):
    '''
    Minimizes given loss/cost function for given number of iterations 
    using lr = 0.1 and 1.0 and prints results
    plots path to computed minimum
    plots contour and surface over given window (xmin,xmax),(ymin,ymax)
    plots gradient surfaces with respect to theta_0 and theta_1
    '''
    print('theta_0_lcm ', theta_0_lcm)
    
    # compute minimizing theta with lr=0.1
    lr = 0.1
    theta_lcm, J_hist_lcm, theta_hist_lcm = gradientDescent(costFunc,X_train_o,y_train,theta_0_lcm,\
                                                                        lr, iterations)
    print('Minimizing theta (lr = 0.1)', theta_lcm)
    print('Loss value (lr = 0.1) ', J_hist_lcm[-1])
    print('\n')
    
    # compute minimizing theta with lr=1.0
    lr = 1.0
    theta_lr1_lcm, J_hist_lr1_lcm, theta_hist_lr1_lcm = gradientDescent(costFunc, \
                                                                        X_train_o,y_train,theta_0_lcm, \
                                                                        lr,iterations)
    
    print('Minimizing theta (lr = 1.0) ', theta_lr1_lcm)
    print('Loss value (lr = 1.0) ', J_hist_lr1_lcm[-1])

    # Convergence path: selection of few points, first 20 every 400, 
    #remaining every 4000 iterations
    # path of GD with lr = .1
    L = len(theta_hist_lcm)
    #GD may stop early in case of no improvement beyond machine epsilon
    if L > 8000: 
        th_lcm_short_list = [theta_hist_lcm[i] for i in range(0,8000,400)]
        [th_lcm_short_list.append(theta_hist_lcm[i]) for i in range(8000,L,4000)]
        
        #corresponding loss value J_hist_lr1_lcm
        J_lcm_short_list= [J_hist_lcm[i] for i in range(0,8000,400)]
        [J_lcm_short_list.append(J_hist_lcm[i]) for i in range(8000,L,4000)]
    else: 
        th_lcm_short_list = [theta_hist_lcm[i] for i in range(0,L,400)]
        J_lcm_short_list= [J_hist_lcm[i] for i in range(0,L,400)]
        
    # To plot path; store theta_0 and theta_1 values in seperate 
    # lr = 0.1
    ths11_lcm = []
    ths22_lcm = []
    
    
    for i in range(len(th_lcm_short_list)):
        ths11_lcm.append(th_lcm_short_list[i][0])
        ths22_lcm.append(th_lcm_short_list[i][1])
    
   
    # to plot path on gradient surface
    # initialiaze gradient arrays
    grads_theta_0 = np.zeros(len(ths11_lcm))
    grads_theta_1 = np.zeros(len(ths11_lcm))
    # compute gradients for corresponding thetas on path
    for i in range(len(ths11_lcm)):
        theta = np.array([ths11_lcm[i], ths22_lcm[i]])
        J, grads = costFunc(theta, X_train_o,y_train)
        grads_theta_0[i] = grads[0]
        grads_theta_1[i] = grads[1]
        
    #path of DG with lr = 1.0
    L = len(theta_hist_lr1_lcm)
    #GD may stop early in case of no improvement beyond machine epsilon
    if L > 8000: 
        th_lcm_lr1_short_list = [theta_hist_lr1_lcm[i] for i in range(0,8000,500)]
        [th_lcm_lr1_short_list.append(theta_hist_lr1_lcm[i]) for i in range(8000,L,4000)]
        
        #corresponding loss value J_hist_lr1_lcm
        J_lcm_lr1_short_list= [J_hist_lr1_lcm[i] for i in range(0,8000,500)]
        [J_lcm_lr1_short_list.append(J_hist_lr1_lcm[i]) for i in range(8000,L,4000)]
    else: 
        th_lcm_lr1_short_list = [theta_hist_lr1_lcm[i] for i in range(0,L,500)]
        J_lcm_lr1_short_list= [J_hist_lr1_lcm[i] for i in range(0,L,500)]
        
    
    # To plot path; store theta_0 and theta_1 values in seperate 
    # lr = 1.0
    ths11_lcm_lr1 = []
    ths22_lcm_lr1 = []

    for i in range(len(th_lcm_lr1_short_list)):
        ths11_lcm_lr1.append(th_lcm_lr1_short_list[i][0])
        ths22_lcm_lr1.append(th_lcm_lr1_short_list[i][1])
        
    #to plot path on gradient surface
    # initialiaze gradient arrays
    grads_lr1_theta_0 = np.zeros(len(ths11_lcm_lr1))
    grads_lr1_theta_1 = np.zeros(len(ths11_lcm_lr1))
    # compute gradients for corresponding thetas on path
    for i in range(len(ths11_lcm_lr1)):
        theta = np.array([ths11_lcm_lr1[i], ths22_lcm_lr1[i]])
        J, grads = costFunc(theta, X_train_o,y_train)
        grads_lr1_theta_0[i] = grads[0]
        grads_lr1_theta_1[i] = grads[1]
    
    #plot selected paths overlaied
    plt.plot(ths11_lcm_lr1, ths22_lcm_lr1, '-o')
    plt.plot(ths11_lcm, ths22_lcm, '-*')
    plt.legend(['lr = 1.0','lr = 0.1'])
    plt.title('Path of theta values: \n first 20 points every 400 steps then every 4000 steps')

    # Prepare data for contour and surface plots
    tt1_lcm = np.sort(np.concatenate((np.linspace(xmin,xmax,401),np.array(ths11_lcm))))
    tt2_lcm = np.sort(np.concatenate((np.linspace(ymin,ymax,401),np.array(ths22_lcm))))
    
    # make grid
    tt12_lcm, tt21_lcm = np.meshgrid(tt1_lcm, tt2_lcm)
    
    #initialize loss and gradient on top of the grid
    K12_lcm = np.zeros((tt12_lcm.shape[0],tt21_lcm.shape[0]))
    grad12_th1 = np.zeros((tt12_lcm.shape[0],tt21_lcm.shape[0]))
    grad12_th2 = np.zeros((tt12_lcm.shape[0],tt21_lcm.shape[0]))
    
    # compute loss and gradient for each theta on the grid
    for i in range(tt12_lcm.shape[0]):
        for j in range(tt21_lcm.shape[0]):
            theta = np.array([tt12_lcm[i,j], tt21_lcm[i,j]])
            K12_lcm[i,j], grad= costFunc(theta, X_train_o, y_train)
            grad12_th1[i,j] = grad[0]
            grad12_th2[i,j] = grad[1]
    
    #Surface plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(tt12_lcm, tt21_lcm, K12_lcm, cmap='viridis')
    ax.plot(ths11_lcm, ths22_lcm, J_lcm_short_list, c='g', linewidth = 3)
    ax.plot(ths11_lcm_lr1, ths22_lcm_lr1, J_lcm_lr1_short_list, c='r', linewidth = 3)
    ax.view_init(30,140)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.title('Surface')
    
    # contour plot
    ax = plt.subplot(222)
    plt.contour(tt1_lcm, tt2_lcm, K12_lcm, linewidths=2, cmap='viridis', levels = 30)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.plot(ths11_lcm, ths22_lcm, 'g-*')
    plt.plot(ths11_lcm_lr1, ths22_lcm_lr1, 'r-o')
    plt.plot(gmin[0],gmin[1], 'ro') 
    plt.plot(theta_lcm[0], theta_lcm[1], 'm*')
    #plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
    plt.title('Contour, showing thetas leading to minimum')
    
    if plot_gradient: 
        #plot gradient with respect to theta_0
        ax = fig.add_subplot(223, projection='3d')
        ax.plot_surface(tt12_lcm, tt21_lcm, grad12_th1, cmap='viridis')
        ax.plot(ths11_lcm, ths22_lcm, grads_theta_0, c='g', linewidth = 3) 
        ax.plot(ths11_lcm_lr1, ths22_lcm_lr1, grads_lr1_theta_0, c='r', linewidth = 3)
        ax.view_init(30,140)
        plt.xlabel('theta_0')
        plt.ylabel('theta_1')
        plt.title('gradient with respect to theta_0')

        #plot gradient with respect to theta_1
        ax = fig.add_subplot(224, projection='3d')
        ax.plot_surface(tt12_lcm, tt21_lcm, grad12_th2, cmap='viridis')
        ax.plot(ths11_lcm, ths22_lcm, grads_theta_1, c='g', linewidth = 3)
        ax.plot(ths11_lcm_lr1, ths22_lcm_lr1, grads_lr1_theta_1, c='r', linewidth = 3)
        ax.view_init(30,140)
        plt.xlabel('theta_0')
        plt.ylabel('theta_1')
        plt.title('gradient with respect to theta_1')
    plt.suptitle('Loss function surface and contour ')
    plt.show()
    

