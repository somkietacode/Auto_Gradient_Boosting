import numpy as np
from Awesome_Linear_Regression import linearregression as LR

class AGB:
  
  def __init__(self,x_training_data,y_training_data,learning_rate):
    
    self.x_training_data,self.y_training_data = x_training_data,y_training_data
    self.y_mean = y_training_data.mean()
    self.learning_rate = learning_rate
    self.y_to_mean = y_training_data - self.y_mean
    optimized = False
    self.lrs = []
    while optimized == False :
      node = LR(self.x_training_data,self.y_to_mean)
      beta,rss = node.leastsquare()
      self.lrs.append(node)
      predicted_residual = np.transpose([np.apply_along_axis(node.predict, axis=1, arr=x_training_data)])
      self.y_to_mean = self.y_to_mean - (predicted_residual * self.learning_rate)
      if  self.y_to_mean.mean()< self.y_mean  :
          if  np.log(np.sqrt(np.mean(np.square(self.y_to_mean)))) <= 0 :
              optimized = True
      if self.y_mean <  self.y_to_mean.mean()   :
        if  0 <= np.log(np.sqrt(np.mean(np.square(self.y_to_mean))))  :
              optimized = True

  def predict(self,x):
    prediction = self.y_mean
    for node in self.lrs :
        prediction += node.predict(x) * self.learning_rate
    return prediction


if __name__ == "__main__" :
  x = np.matrix([[0,1],[7,50],[30,40],[50,23]])
  y = np.matrix([[1],[57],[70],[73]])
  AGB = AGB(x,y,0.00001)
  print(AGB.predict(np.matrix([[0,1]])))
