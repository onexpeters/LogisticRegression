cimport numpy as np
class LogisticRegression:
    def initial_Weights(self,features): 
        extra_column=np.ones((len(features),1))#add an extra columns on feature matrix for the bias
        np.concatenate((extra_column, features), axis=1)
        weights=np.random.randn(1,len(features[0]))#Genarate the initial weights
        return weights
    def compute_z(self,weights,features):#compute z values
        z=weights[0][0:1]+np.dot(weights[0][1:],features[:,1:].transpose())
        return z
    def estimate_of_y_via_sigmoid(self,z):#estimate the target i.e y_estimate
        y_estimate=1/(1+np.exp(-z))
        return y_estimate
    def derivative(self,y_estimate,y,features):#derivative of Logistic cost function i.e gradient_descent
        derivatives=np.dot((y_estimate-y).transpose(),features)
        derivatives=derivatives/len(features)
        return derivatives
    def adjusting_weights(self,weights,derivative,rate):#convergence of weights
        weights= weights-(rate*derivative)
        return weights
    def predict_category(self,y_estimate,threshold):#classification 
        predict=[]
        for i in y_estimate:
            if(i<=threshold):
                predict.append(0)
            elif(i>threshold):
                predict.append(1)
        return predict
    def accuracy(self,predicted_labels, actual_labels):
            diff = predicted_labels - actual_labels
            return 1.0 - (float(np.c
        
        
