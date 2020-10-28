
    # Comparing Cross Entropy and Mean Squared Error for Classification Loss
   
    Using a simple case of binary classification involving one feature, we look at the performace and visualize the cross entropy loss and the mean squared error loss. 
    We find that the mean square loss is non-convex, the optimizing algorithm converges to the closest minimum that maybe local and not the global one. 
    
    ![MSE loss](images/MSE_loss.png)
    
    To facilitate visualization of the cross entropy (CE) loss, I extend it range of permissible values of the sigmoid function into the CE loss so that it always lies between 1 - 1e-10 and 0 + 1e-10. 
    On the original range of sigmoid values the CE loss is convex and has one global minimum.
    
    ![CE loss](images/CE_loss.png)
    
    However, if we start with the correct initial parameters, we get similar classification performance. 
