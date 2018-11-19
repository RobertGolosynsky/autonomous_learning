import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.stats import norm

# Create a surface plot and projected filled contour plot under it.
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

def load_data_1D():
    """
    Get an artifical data set containing samples of data for 2 different normal distributed class models.

    Returns: (class 0, class 1)
    ========
    (c0, c1)
    """
    np.random.seed(564783)
    return np.hstack((np.random.normal(size=200,loc=0.3,scale=0.05), np.random.normal(size=100,loc=0.7,scale=0.1)))

def get_2_class_data_1D():
    """
    Get an artifical data set containing samples of data for 2 different normal distributed class models.

    Returns: (class 0, class 1)
    ========
    (c0, c1)
    """
    np.random.seed(5647839)
    return (np.random.normal(size=200,loc=0.3,scale=0.05), np.random.normal(size=50,loc=0.7,scale=0.1))

def get_cancer():
	"""
	Get an artifical data set containing samples of patients without and with cancer.

	Returns:
	========
	(nocancer, cancer)
	"""
	np.random.seed(5647839)
	return (np.random.normal(size=200,loc=7.0,scale=2), np.random.normal(size=50,loc=10.5,scale=1))
	
def get_2_class_data_2D():
	"""
	Get an artiicial 2D data set containing samples of Data for 2 different normal distributed class models.
	
	Returns: (class 0_x, class 0_y, class 1_x, class_1_y)
	========
	(c0_x, c0_y, c1_x, c1_y)
	"""
	np.random.seed(5647839)
	x1, y1 = np.random.normal(size=50, loc=0.9, scale=0.2), np.random.normal(size=50, loc=0.2, scale=0.5)
	np.random.seed(852962)
	x2, y2 = np.random.normal(size=50, loc=0.3, scale=0.1), np.random.normal(size=50, loc=0.5, scale=0.3)
	
	c0, c1 = np.array([x1,y1]), np.array([x2,y2])
	t_0, t_1 = np.zeros(50), np.ones(50)
    
	X = np.column_stack((c0,c1)).T
	t= np.hstack((t_0,t_1))
	
	return X, t
	
def plot_hist_1D(C, binwidth):
	nullfmt = NullFormatter()         # no labels
	x=np.linspace(0,1,1000)

	# definitions for the axes
	left, width = 0.1, 0.7
	bottom, height = 0.1, 0.7
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]

	# start with a rectangular Figure
	plt.figure(1, figsize=(8, 8))
	axScatter = plt.axes(rect_scatter)
	axHistx = plt.axes(rect_histx)

	# no labels
	axHistx.xaxis.set_major_formatter(nullfmt)

	# the scatter plot:
	axScatter.scatter(C, np.zeros(len(C)), marker='x', c='k')
	axScatter.set_xlim((0, 1))

	bins = np.arange(0, 1 + binwidth, binwidth)
	axHistx.hist(C, bins=bins, normed=True)
	axHistx.set_xlim(axScatter.get_xlim())

	plt.show()
    
def kernel_plot(px):
	#c0, c1 = get_2_class_data_1D()
	nullfmt = NullFormatter()         # no labels
	x = np.linspace(0,1,1000)

	# definitions for the axes
	left, width = 0.1, 0.7
	bottom, height = 0.1, 0.7
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]

	# start with a rectangular Figure
	plt.figure(1, figsize=(12, 12))

	axHistx = plt.axes(rect_histx)

	# no labels
	axHistx.xaxis.set_major_formatter(nullfmt)

	density = (norm.pdf(x, 0.3, 0.05) + norm.pdf(x, 0.7, 0.1))/np.sqrt(sum(norm.pdf(x, 0.7, 0.1))**2 + sum(norm.pdf(x, 0.3, 0.05))**2)
	axHistx.plot(x,density)
	axHistx.plot(x,px/sum(px))

	plt.show()
