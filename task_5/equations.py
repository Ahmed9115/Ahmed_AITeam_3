import numpy as np 

class regresion:
    def normalization(self,x_train,x_valid ,x_test,y_train,y_valid ,y_test):
        xmin = np.min(x_train,axis=0)
        xmax = np.max(x_train,axis=0) 
        ymin = np.min(y_train,axis=0)
        ymax = np.max(y_train,axis=0)

        x_train =  (x_train - xmin)/( xmax- xmin)
        x_valid = (x_valid - xmin)/( xmax - xmin)
        x_test  = (x_test -xmin)/( xmax - xmin)

        y_train =  (y_train - ymin)/( ymax - ymin)
        y_valid =  (y_valid - ymin)/(ymax - ymin)
        y_test =  (y_test - ymin)/( ymax - ymin)
        
        return x_train,x_valid ,x_test , y_train,y_valid ,y_test



    def normal_equation(self,xtrain ,ytrain ):
        xtrans = xtrain.T
        temp = xtrans@xtrain
        temp = np.linalg.inv(temp)
        final = temp@xtrans@ytrain
        return final




    def MSE(self,x,y,random =True):
        if random :
            rand = np.random.randn(x.shape[1] , 1)
            pred = x@rand
            mse = np.mean((pred - y)**2)
            print(mse)

        else :
        
            pred = x
            mse = np.mean((pred - y)**2)
            print(mse)




    def MAE(self,x,y,random =True):
         if random :
            rand = np.random.randn(x.shape[1] , 1)
            pred = x@rand
            mae = np.mean(np.absolute((pred - y)))
            print(mae)
         else :
            pred = x
            mae = np.mean(np.absolute((pred - y)))
            print(mae)




    def gradient_descent_with_mean_squared(self,x, y , iterations , learning_rate): 
        weight = np.random.randn(x.shape[1] , 1)
        for i in range(iterations):
            assumption = x@weight
            weight -=(learning_rate/len(x))*(x.T@(assumption - y))
        return weight

    def gradient_descent_with_mean_absolute(self,x, y , iterations , learning_rate): 
        weight = np.random.randn(x.shape[1] , 1)
        for i in range(iterations):
            assumption = x@weight
            weight -=(learning_rate/len(x))*(x.T@np.sign(assumption - y))  # i didnt find any reference explaining the idea so i asked ai about it !!
        return weight