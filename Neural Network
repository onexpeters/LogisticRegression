"This a a neural network implementation based on the discussion nielsen Michael book"
import numpy as np
class Neural_Network:
    "Generate initial weights and biases"
    def generate_initial_weights(self,neurons_per_layer):
        weights=[]
        biases=[]
        for j,k in zip(neurons_per_layer[1:],neurons_per_layer[:-1]):
            weights.append(np.random.randn(j,k))
            biases.append(np.random.randn(j,1))
        return weights,biases
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    def feedforward(self,activation,weights,biases):
        activations=[activation]
        z_s=[]
        for weight,bias in zip(weights,biases):
            z=np.dot(weight,activation)+bias
            activation=self.sigmoid(z)
            z_s.append(z)
            activations.append(activation)
        return activations,z_s
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1.0-self.sigmoid(z))
    "Read page 58 from equation 29"
    def deltas_(self,z_s,activations,expected,weights):
        deltas=[]
        delta=(activations[-1]-expected)*self.sigmoid_prime(z_s[-1])
        deltas.append(delta)
        for weight_,z_i in zip(weights[::-1],z_s[-2::-1]):
            delta=np.dot(weight_.transpose(),delta)*self.sigmoid_prime(z_i)
            deltas.append(delta)
        return deltas
    "This is according to BP4 PG 60"
    def weightsAndBiasderivatives(self,deltas,activations):
        weight_derivatives=[]
        bias_derivatives=[]
        for a,d in zip(activations[-2::-1],deltas):
            weight_derivative=np.dot(d,a.transpose())
            bias_derivative=d
            weight_derivatives.append(weight_derivative)
            bias_derivatives.append(bias_derivative)
        return weight_derivatives,bias_derivatives
    "Gradient descend i.e  stochastic gradient descend"
    def adjust_weightsAndBiases(self,weights,biases,weight_derivatives,bias_derivatives,rate,n):
        weights_derivatives=[(rate*x)/n for x in weight_derivatives]
        bias_derivatives=[(rate*x)/n for x in bias_derivatives]
        weights=[weight-w_derivative for weight,w_derivative in zip(weights,weight_derivatives)]
        biases=[bias-w_bias for bias,w_bias in zip(biases,bias_derivatives)]
        return weights,biases
         "Network Performance"
    def evaluate(self,test_data,weights,biases):
        prediction_results=[]
        for x,y in test_data:
            activations,z_s=self.feedforward(x,weights,biases)
            prediction=np.argmax(activations[-1])
            prediction_results.append(prediction)
            expected=[y for x,y in test_data]
            final_test_results = [(x, y)for (x, y) in zip(prediction_results,expected)]
        return sum(int(x == y) for (x, y) in final_test_results)
    " Implementation function"
    def AggregatedFunction(self,mini_batch,weights,biases,rate):
        bias_derivatives = [np.zeros(b.shape) for b in biases]
        weight_derivatives = [np.zeros(w.shape) for w in weights]
        for x,y in mini_batch:
            activations,z_s =self.feedforward(x,weights,biases)
            deltas=self.deltas_(z_s,activations,y,weights)
            weight_derivative,bias_derivative=self.weightsAndBiasderivatives(deltas,activations)
            weight_derivatives=[w_d+w for w_d,w in zip(weight_derivatives,weight_derivative[::-1])]
            bias_derivatives=[b_d+b for b_d,b in zip(bias_derivatives,bias_derivative[::-1])]
            weight_derivative=[]
            bias_derivative=[]
        weights,biases=self.adjust_weightsAndBiases(weights,biases,weight_derivatives,bias_derivatives,rate,len(mini_batch))
        return weights,biases
    "uplifted from the book with slight modifications to fit this code"
    def SGD(self,training_data,epochs,mini_batch_size,neurons_per_layer,rate,test_data=None):
        weights,biases=self.generate_initial_weights(neurons_per_layer)
        if test_data:n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                weights,biases=self.AggregatedFunction(mini_batch,weights,biases,rate)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data,weights,biases),n_test))
            else:
