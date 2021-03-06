class Regression_descent:
    def generateWeightsAndBiases(self,features):#initial weights and bias
        np.random.seed(43)
        w=np.random.randn(len(features[0]))
        b=np.random.randn(1,1)
        return w.reshape(-1,1),b.reshape(-1,1)
    def GenerateY(self,w,x,b):#compute Y estimate
        y=np.dot(w.transpose(),x.transpose())+b
        return y.reshape(-1,1)
    def cost(self,y,t):#Compute error term
        error=(y-t)
        return error
    def WeightDerivativePlusrate(self,features,error,rate):#Derivative multiplied by rate
        weight_derivative=(np.dot(features.T,error))/len(features)
        derivative=rate*(weight_derivative) 
        return derivative
    def adjusted_weights(self,weights,derivative):#adjust weight 
        return weights-derivative
    def adjusted_biases(self,biases,error,features,rate):#adjust bias 
        error_bias=error.sum()/len(features)
        return (biases-(rate*error_bias))
