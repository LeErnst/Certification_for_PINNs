import torch as tr
from math import sqrt

######################### quadrature weights and points ########################


############## quadrature 1D
### gauß-legendre quadrature 1D
# n = 1:
gleg_alpha_1 = tr.tensor([2], dtype=tr.double)
gleg_x_1     = tr.tensor([0], dtype=tr.double)
# n = 2:
gleg_alpha_2 = tr.ones(2, dtype=tr.double)
gleg_x_2     = tr.tensor([-sqrt(1/3), sqrt(1/3)], dtype=tr.double)
# n = 3:
gleg_alpha_3 = tr.tensor([5/9, 8/9, 5/9], dtype=tr.double)
gleg_x_3     = tr.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=tr.double)
# n = 5:
gleg_alpha_5 = tr.tensor([(322-13*sqrt(70))/900,\
                          (322+13*sqrt(70))/900,\
                           128/255, \
                          (322+13*sqrt(70))/900,\
                          (322-13*sqrt(70))/900], dtype=tr.double)
gleg_x_5     = tr.tensor([-1/3*sqrt(5+2*sqrt(10/7)), \
                          -1/3*sqrt(5-2*sqrt(10/7)), \
                           0, \
                           1/3*sqrt(5-2*sqrt(10/7)), \
                           1/3*sqrt(5+2*sqrt(10/7))], dtype=tr.double)

# n = 10:
gleg_alpha_10 = tr.tensor([0.0666713443086881,\
                           0.1494513491505806,\
                           0.2190863625159820,\
                           0.2692667193099963,\
                           0.2955242247147529,\
                           0.2955242247147529,\
                           0.2692667193099963,\
                           0.2190863625159820,\
                           0.1494513491505806,\
                           0.0666713443086881], dtype=tr.double)

gleg_x_10     = tr.tensor([-0.9739065285171717,\
                           -0.8650633666889845,\
                           -0.6794095682990244,\
                           -0.4333953941292472,\
                           -0.1488743389816312,\
                            0.1488743389816312,\
                            0.4333953941292472,\
                            0.6794095682990244,\
                            0.8650633666889845,\
                            0.9739065285171717], dtype=tr.double)

### gauß-lobatto quadrature 1D
# n = 2:
glob_alpha_2 = tr.tensor([1, 1], dtype=tr.double)
glob_x_2     = tr.tensor([-1, 1], dtype=tr.double)

# n = 3:
glob_alpha_3 = tr.tensor([1/3, 4/3, 1/3], dtype=tr.double)
glob_x_3     = tr.tensor([-1, 0, 1], dtype=tr.double)

# n = 4:
glob_alpha_4 = tr.tensor([1/6, 5/6, 5/6, 1/6], dtype=tr.double)
glob_x_4     = tr.tensor([-1, -sqrt(1/5), sqrt(1/5), 1], dtype=tr.double)

# n = 5:
glob_alpha_5 = tr.tensor([1/10, 49/90, 32/45, 49/90, 1/10], dtype=tr.double)
glob_x_5     = tr.tensor([-1, -sqrt(3/7), 0, sqrt(3/7), 1], dtype=tr.double)

# n = 6:
glob_alpha_6 = tr.tensor([1/15, (14-sqrt(7))/30, \
                                (14+sqrt(7))/30, \
                                (14+sqrt(7))/30, \
                                (14-sqrt(7))/30, 1/15], dtype=tr.double)
glob_x_6     = tr.tensor([-1, -sqrt(1/3+(2*sqrt(7))/21), \
                              -sqrt(1/3-(2*sqrt(7))/21), \
                               sqrt(1/3-(2*sqrt(7))/21), \
                               sqrt(1/3+(2*sqrt(7))/21), 1], dtype=tr.double)

# n = 7:
glob_alpha_7 = tr.tensor([1/21, (124-7*sqrt(15))/350, \
                                (124+7*sqrt(15))/350, \
                                    256/525, \
                                (124+7*sqrt(15))/350, \
                                (124-7*sqrt(15))/350, 1/21], dtype=tr.double)
glob_x_7     = tr.tensor([-1, -sqrt(5/11+(2/11)*sqrt(5/3)), \
                              -sqrt(5/11-(2/11)*sqrt(5/3)), \
                                        0, \
                               sqrt(5/11-(2/11)*sqrt(5/3)), \
                               sqrt(5/11+(2/11)*sqrt(5/3)), 1], dtype=tr.double)


