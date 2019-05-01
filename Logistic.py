class regression:
    def sigmoid(self,z):
        return 1/(1.0+np.exp(-z))
    def initial_theta(self,array):#Initial weights generation
        theta_note=np.random.rand(1,1)
        thetas=np.random.rand(len(array[0]),1)
        return thetas,theta_note
    def compute_z(self,thetas,theta_note,x):#Computing Z
        z=[]
        total=0
        for j in range(len(x)):
            for i in range(len(thetas)):
                total=total+(thetas[i]*x[j][i])
            total=total+theta_note
            z.append(total)
            total=0
        return z
    def prediction(self,z):
        prediction=[]
        for i in range(len(z)):
            prediction.append(self.sigmoid(z[i]))
        return prediction
    def update_theta_note(self,theta_note,array,z,y,factor):#Update theta_0
        total_note=0
        for i in range(len(array)):
            total_note=total_note+(self.sigmoid(z[i])-y[i])
        theta_note=theta_note-(total_note/len(array)*factor)
        return theta_note
    def update_theta(self,theta,array,z,y,factor):#Update theta1,theta2...
        theta_update=[]
        total=0
        for j in range(len(theta)):
            for i in range(len(array)):
                total=total+(self.sigmoid(z[i])-y[i])*array[i][j]
            theta_update.append(factor*total/len(array))
            total=0
        return theta-theta_update
    def cost_function(self,z,labels):#cost function
        cost1=0
        cost2=0
        for  i in range(len(labels)):
            cost1=cost1+(labels[i]*np.log(self.sigmoid(z[i])))
            cost2=cost2+((1-labels[i])*np.log(1-self.sigmoid(z[i])))
        return -(cost1+cost2)/len(labels)
    def accuracy(self,z,labels):
        diff=0
        for i in range(len(z)):
            diff =diff+np.abs((z[i]-labels[i]))
        return 1.0-(float(np.count_nonzero(diff)) / len(diff))