import math
import scipy.special
import mpmath

def gammaPlus(a,x):
	return scipy.special.gammaincc(a, x) * gamma(x)

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

def exactLimit(k, alpha, nu):
	""""""
	if alpha == 1:
		print("alpha == 1")
	xi = (4*alpha*nu) / (math.pi*(2*alpha -1))
	limit = 1 / (8*alpha*(alpha-1)*gammaPlus(k-2*alpha, xi)) * (-gammaPlus(k-2*alpha, xi) - 2*(alpha*(alpha-0.5)**2*xi**2 * gammaPlus(k-2*alpha-2, xi) ) / (alpha-1) + 8*alpha*(alpha-0.5)*xi*gammaPlus(k-2*alpha-1,xi) + 4*xi**(4*alpha-2) * gammaPlus(k-6*alpha+2,xi)*((2**(-4*alpha) * (3*alpha-1))/(alpha-1) + (alpha-0.5)*betaMinus(0.5,1+2*alpha,-2+2*alpha)) + xi**(k-2*alpha)*gamma(2*alpha+1)*math.exp(-xi)*U(2*alpha+1, 1+k-2*alpha,xi) - xi**(4*alpha-2)*gamma(2*alpha+1)*G(xi,alpha,k))

	print("*******")
	print(1 / (8*alpha*(alpha-1)*gammaPlus(k-2*alpha, xi)))
	print((-gammaPlus(k-2*alpha, xi) - 2*(alpha*(alpha-0.5**2)*xi**2 * gammaPlus(k-2*alpha-2, xi) ) / (alpha-1) ))
	print(8*alpha*(alpha-0.5)*xi*gammaPlus(k-2*alpha-1,xi))
	print(4*xi**(4*alpha-2) * gammaPlus(k-6*alpha+2,xi)*((2**(-4*alpha) * 3*alpha-1)/(alpha-1)) + (alpha-0.5)*betaMinus(0.5,1+2*alpha,-2+2*alpha))
	print((alpha-0.5)*betaMinus(0.5,1+2*alpha,-2+2*alpha))
	print(betaMinus(0.5,1+2*alpha,-2+2*alpha))
	print(xi**(k-2*alpha)*gamma(2*alpha+1)*math.exp(-xi)*U(2*alpha+1, 1+k-2*alpha,xi))
	print(xi**(4*alpha-2)*gamma(2*alpha+1)*G(xi,alpha,k))
	print(U(2*alpha+1, 1+k-2*alpha,xi))
	print(G(xi,alpha,k))
	return limit

for k in range(2,25+1):
	print(exactLimit(k=k, alpha=0.8, nu=1))


# alpha = 0
# while alpha <=4:
# 	print(str(alpha) + ": " + str((alpha-0.5)*betaMinus(0.5,1+2*alpha,-2+2*alpha)))
# 	alpha += 0.2