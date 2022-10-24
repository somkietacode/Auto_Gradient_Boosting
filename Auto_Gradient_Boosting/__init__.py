import numpy as np
from .linearregression import linearregression

if __name__ == "__main__" :
  x = np.matrix([[0,1],[1,4],[7,8],[50,23]])
  y = np.matrix([[2],[9],[23],[96]])
  Lr = linearregression(x,y)
  Beta, rss = Lr.leastsquare()
  print(Beta)
  print(rss)
  px = np.matrix([[4,7]])
  r_y = Lr.predict(px)
  print(r_y)
