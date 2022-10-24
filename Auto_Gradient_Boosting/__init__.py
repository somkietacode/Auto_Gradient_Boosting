import numpy as np
from .main import AGB

if __name__ == "__main__" :
  x = np.matrix([[0,1],[1,4],[7,8],[50,23]])
  y = np.matrix([[2],[9],[23],[96]])
  agb = AGB(x,y,0.01)
  px = np.matrix([[4,7]])
  r_y = agb.predict(px)
  print(r_y)
