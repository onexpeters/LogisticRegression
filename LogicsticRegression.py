class LogicsticRegression:
    def initial_weights(self,array):
        np.random.seed(43)
        weights=np.random.randn(len(array[0]),1)
        return weights
    def calculate_z(self,features_,weights_):
        z=np.dot(features_,weights_)
        return z
    def sigmoid_(self,z):
        predict=1/(1+np.exp(-z))
        return predict
    def predictions_(self,z):
        predictions=[]
        for i in range(len(z)):
            predictions.append(self.sigmoid_(z[i]))
        return predictions
    def update_of_weights(self,predictions_,labels_,features_,weights_,factor_):
        adjustment=np.dot(features_.transpose(),(predictions_-labels_))
        adjustment=adjustment/len(labels_)
        adjustment=adjustment*factor_
        return (weights_-adjustment)
        
        