import numpy as np
import pandas as pd
import time
from sklearn import datasets
from sklearn.utils import shuffle 
from scipy import optimize
import matplotlib.pyplot as plt


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
        #save results per epoch
        J_hist.append(J)
        theta_hist.append(theta)
        # check theta is improving
        if epoch > 100:
            theta_step = theta_hist[-2] - theta_hist[-1]
        
            # stop if improvement in theta is at machine epsilon scale
            if (abs(theta_step[0]) < 1/(2**50)) & (abs(theta_step[1]) < 1/(2**50)):
                print('early stop due to theta_step close to machine epsilon :', theta_step) 
                return theta, J_hist, theta_hist
        
        
        if epoch % 10000 == 0:
            print('epoch : {},  theta : {},  loss : {:0.8f} '.format(epoch, theta, J))
    return theta, J_hist, theta_hist 


#results for local minimum

def local_min_plots(costFunc,X_train_o, y_train, theta_0_lcm, batch_size, iterations, xmin, xmax, ymin, ymax, gmin, \
                           plot_gradient = True):
    '''
    Minimizes given loss/cost function for given number of iterations 
    using lr = 0.1 and 1.0 and prints results
    plots path to computed minimum
    plots contour and surface over given window (xmin,xmax),(ymin,ymax)
    plots gradient surfaces with respect to theta_0 and theta_1
    '''
    print('theta_0_lcm ', theta_0_lcm)
    
    # compute minimizing theta with lr=1.0
    lr = 1.0
    t0 = time.time()
    theta_lr1_lcm, J_hist_lr1_lcm, theta_hist_lr1_lcm = mini_bGD(costFunc, \
                                                                        X_train_o,y_train,theta_0_lcm, \
                                                                        batch_size, lr,iterations)
    
    print('mini batch GD took {0:4.3f} minutes'.format((time.time() - t0)/60))
    print('Minimizing theta (lr = 1.0) ', theta_lr1_lcm)
    print('Loss value (lr = 1.0) ', J_hist_lr1_lcm[-1])
    
    
    #path of DG with lr = 1.0
    L = len(theta_hist_lr1_lcm)
    M = min(8000,L)
    #GD may stop early in case of no improvement beyond machine epsilon
    if L > 8000: 
        th_lcm_lr1_short_list = [theta_hist_lr1_lcm[i] for i in range(0,M,500)]
        [th_lcm_lr1_short_list.append(theta_hist_lr1_lcm[i]) for i in range(M,L,4000)]
        
        #corresponding loss value J_hist_lr1_lcm
        J_lcm_lr1_short_list= [J_hist_lr1_lcm[i] for i in range(0,M,500)]
        [J_lcm_lr1_short_list.append(J_hist_lr1_lcm[i]) for i in range(M,L,4000)]
    else: 
        th_lcm_lr1_short_list = [theta_hist_lr1_lcm[i] for i in range(0,L,500)]
        J_lcm_lr1_short_list= [J_hist_lr1_lcm[i] for i in range(0,L,500)]
        
    
    # To plot path; store theta_0 and theta_1 values in seperate arrays
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
    plt.legend(['lr = 1.0'])
    plt.title('Path of theta values: \n first 20 points every 400 steps then every 4000 steps')

    t0 = time.time()
    # Prepare data for contour and surface plots
    tt1_lcm = np.sort(np.concatenate((np.linspace(xmin,xmax,401),np.array(ths11_lcm_lr1))))
    tt2_lcm = np.sort(np.concatenate((np.linspace(ymin,ymax,401),np.array(ths22_lcm_lr1))))
    
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
    
    print('Computations for surface plots time {0:4.3f} minutes'.format((time.time()-t0)/60)) 
    #Surface plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(tt12_lcm, tt21_lcm, K12_lcm, cmap='viridis')
    #ax.plot(ths11_lcm, ths22_lcm, J_lcm_short_list, c='g', linewidth = 3)
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
    #plt.plot(ths11_lcm, ths22_lcm, 'g-*')
    plt.plot(ths11_lcm_lr1, ths22_lcm_lr1, 'r-o')
    plt.plot(gmin[0],gmin[1], 'ro') 
    #plt.plot(theta_lcm[0], theta_lcm[1], 'm*')
    #plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
    plt.title('Contour, showing thetas leading to minimum')
    
    if plot_gradient: 
        #plot gradient with respect to theta_0
        ax = fig.add_subplot(223, projection='3d')
        ax.plot_surface(tt12_lcm, tt21_lcm, grad12_th1, cmap='viridis')
        #ax.plot(ths11_lcm, ths22_lcm, grads_theta_0, c='g', linewidth = 3) 
        ax.plot(ths11_lcm_lr1, ths22_lcm_lr1, grads_lr1_theta_0, c='r', linewidth = 3)
        ax.view_init(30,140)
        plt.xlabel('theta_0')
        plt.ylabel('theta_1')
        plt.title('gradient with respect to theta_0')

        #plot gradient with respect to theta_1
        ax = fig.add_subplot(224, projection='3d')
        ax.plot_surface(tt12_lcm, tt21_lcm, grad12_th2, cmap='viridis')
        #ax.plot(ths11_lcm, ths22_lcm, grads_theta_1, c='g', linewidth = 3)
        ax.plot(ths11_lcm_lr1, ths22_lcm_lr1, grads_lr1_theta_1, c='r', linewidth = 3)
        ax.view_init(30,140)
        plt.xlabel('theta_0')
        plt.ylabel('theta_1')
        plt.title('gradient with respect to theta_1')
    plt.suptitle('Loss function surface and contour ')
    plt.show()
    
    
    
    