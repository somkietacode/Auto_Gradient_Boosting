import numpy as np
from Awesome_Linear_Regression import linearregression as LR

class AGB:
  
  def __init__(self,x_training_data,y_training_data):
    
    self.x_training_data,self.y_training_data = x_training_data,y_training_data
    self.y_mean = y_training_data.mean
    self.y_to_mean = y_training_data - self.mean
    optimized = False
    Betas = []
    while optimized == False :
      node = LR(self.x_training_data,self.y_to_mean)
      beta,rss = node.leastsquare()
      Betas.append(beta)
      
      
