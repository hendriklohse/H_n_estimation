import math
import scipy.special
import mpmath

def gammaPlus(a,x):
	return scipy.special.gammaincc(a, x) * gamma(x)

def contfractbeta(x,a,b, ITMAX = 200):

	""" contfractbeta() evaluates the continued fraction form of the incomplete Beta function; incompbeta().
	(Code translated from: Numerical Recipes in C.)"""

	EPS = 3.0e-7
	bm = az = am = 1.0
	qab = a+b
	qap = a+1.0
	qam = a-1.0
	bz = 1.0-qab*x/qap

	for i in range(ITMAX+1):
		em = float(i+1)
		tem = em + em
		d = em*(b-em)*x/((qam+tem)*(a+tem))
		ap = az + d*am
		bp = bz+d*bm
		d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem))
		app = ap+d*az
		bpp = bp+d*bz
		aold = az
		am = ap/bpp
		bm = bp/bpp
		az = app/bpp
		bz = 1.0
		if (abs(az-aold)<(EPS*abs(az))):
			return az

	print('a or b too large or given ITMAX too small for computing incomplete beta function.')

def incompbeta(x,a,b):

	''' incompbeta(a,b,x) evaluates incomplete beta function, here a, b > 0 and 0 <= x <= 1. This function requires contfractbeta(a,b,x, ITMAX = 200)
	(Code translated from: Numerical Recipes in C.)'''

	if (x == 0):
		return 0;
	elif (x == 1):
		return 1;
	else:
		lbeta = math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1-x)
		if (x < (a+1) / (a+b+2)):
			return math.exp(lbeta) * contfractbeta(a, b, x) / a;
		else:
			return 1 - math.exp(lbeta) * contfractbeta(b, a, 1-x) / b;

def betaMinus(x,a,b):
	return scipy.special.betainc(a,b,x) * beta(a,b)

def gamma(x):
	return scipy.special.gamma(x)

def beta(a,b):
	return scipy.special.beta(a,b)

def U(a,b,x):
	return scipy.special.hyperu(a,b,x)

def G(z,alpha,k):
	A1 = []
	A2 = [1,3-2*alpha]
	B1 = [3-4*alpha,-6*alpha+k+2,0]
	B2 = []
	return mpmath.meijerg([A1,A2], [B1,B2],z)

# def exactLimit(k, alpha, nu):
# 	""""""
# 	if alpha == 1:
# 		print("alpha == 1")
# 	xi = (4*alpha*nu) / (math.pi*(2*alpha -1))
# 	limit = 1 / (8*alpha*(alpha-1)*gammaPlus(k-2*alpha, xi)) * (-gammaPlus(k-2*alpha, xi) - 2*(alpha*(alpha-0.5)**2*xi**2 * gammaPlus(k-2*alpha-2, xi) ) / (alpha-1) + 8*alpha*(alpha-0.5)*xi*gammaPlus(k-2*alpha-1,xi) + 4*xi**(4*alpha-2) * gammaPlus(k-6*alpha+2,xi)*((2**(-4*alpha) * (3*alpha-1))/(alpha-1) + (alpha-0.5)*betaMinus(0.5,1+2*alpha,-2+2*alpha)) + xi**(k-2*alpha)*gamma(2*alpha+1)*math.exp(-xi)*U(2*alpha+1, 1+k-2*alpha,xi) - xi**(4*alpha-2)*gamma(2*alpha+1)*G(xi,alpha,k))
#
# 	print("*******")
# 	print(1 / (8*alpha*(alpha-1)*gammaPlus(k-2*alpha, xi)))
# 	print((-gammaPlus(k-2*alpha, xi) - 2*(alpha*(alpha-0.5**2)*xi**2 * gammaPlus(k-2*alpha-2, xi) ) / (alpha-1) ))
# 	print(8*alpha*(alpha-0.5)*xi*gammaPlus(k-2*alpha-1,xi))
# 	print(4*xi**(4*alpha-2) * gammaPlus(k-6*alpha+2,xi)*((2**(-4*alpha) * 3*alpha-1)/(alpha-1)) + (alpha-0.5)*betaMinus(0.5,1+2*alpha,-2+2*alpha))
# 	print((alpha-0.5)*betaMinus(0.5,1+2*alpha,-2+2*alpha))
# 	print(betaMinus(0.5,1+2*alpha,-2+2*alpha))
# 	print(xi**(k-2*alpha)*gamma(2*alpha+1)*math.exp(-xi)*U(2*alpha+1, 1+k-2*alpha,xi))
# 	print(xi**(4*alpha-2)*gamma(2*alpha+1)*G(xi,alpha,k))
# 	print(U(2*alpha+1, 1+k-2*alpha,xi))
# 	print(G(xi,alpha,k))
# 	return limit
#
# for k in range(2,25+1):
# 	print(exactLimit(k=k, alpha=0.8, nu=1))


# alpha = 0
# while alpha <=4:
# 	print(str(alpha) + ": " + str((alpha-0.5)*betaMinus(0.5,1+2*alpha,-2+2*alpha)))
# 	alpha += 0.2

# alpha = 3/4
# nu = 1
# xi = (4*alpha*nu) / (math.pi*(2*alpha -1))
# print(incompbeta(0.5, 2*alpha +1, 2*alpha-2))
# print(beta(2*alpha, 3*alpha - 4))
# print(((3*alpha - 1)/(2**(4*alpha+1)*alpha*(alpha-1)**2) + ((alpha-0.5)*incompbeta(0.5, 2*alpha +1, 2*alpha-2))/(2*(alpha-1)*alpha) - (beta(2*alpha, 3*alpha - 4))/(4*(alpha-1)))*xi**(4*alpha - 2))