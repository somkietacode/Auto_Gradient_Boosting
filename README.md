# Auto_Gradient_Boosting
Implementation of auto gradient boosting algorithm
This repository hosts the development of the Auto_Gradient_Boosting.

## About Awesome Linear Regression

Auto_Gradient_Boosting , is a mathematics API written in Python.
It was developed with a focus on enabling fast experimentation.
*Being able to go from idea to result as fast as possible is key to doing good research.*

Auto_Gradient_Boosting is:

-   **Simple** 
-   **Flexible** 
-   **Powerful** 

## First contact with Awesome Linear Regression

The core data structures of Auto_Gradient_Boosting are __consign__ and __result__.

For installation run :

```
pip install Auto-Gradient-Boosting

```

Here is an `exemple` :

```python
import numpy as np
from Auto_Gradient_Boosting import AGB


# Sample training data set

x = np.matrix([[0,1],[1,4],[7,8],[50,23]])
y = np.matrix([[2],[9],[23],[96]])

# Train the model
learning_rate = 0.01
agb = AGB(x,y,learning_rate)


```

Let make prediction

```python

 px = np.matrix([[4,7]])
 print(agb.predict(px))

```

---
## Support

You can ask questions and join the development discussion:

- [Facebook page](https://www.facebook.com/globalanalysistech) .

---
