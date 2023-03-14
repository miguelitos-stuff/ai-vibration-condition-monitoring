#File to read all the mat files
import matplotlib.pyplot as plt
import scipy.io

#Use a for loop to make lists for every sensor on every gearbox called "(D/H)(1-10)AN(1-10)_list"
#Look at week 1 AI course assignment for inspiration on how to create a for loop with several path names
for i in range(1,10):
    for j in range(3,10):
        pathname= 'data\Damaged\D' + str(i) + '\AN' + str(j)
        print(pathname)
#listname = scipy.io.loadmat(pathname)
