Below are some key points of the algorithm.

Equation: (AEu_x + x)_x = 0

BCs: u(0) = 0
     u_x(1) = 1

1. The training data (input of NN) is not important for this case as it is not a data driven approach.
2. Two NNs are used, one for each bar cross-sectional area, A_left, A_right.
3. Two additional boundary were implemented for the interface of discontinuity d

     u_left(d) = u_right(d)
     u_x_left(d) = (A_left/A_right) * u_x_right(d)

4. Both NNs were trained together (each epoch updates both NN training variables)
