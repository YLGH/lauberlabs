from scipy.constants import mu_0, pi
import numpy as np
from math import cos, sin
r = 1
mu_0


import matplotlib.pyplot as plt

class Point:
	def __init__(self, coordArray):
		self.coordinates = coordArray
	def __str__(self):
		return str(self.coordinates)

class Vector:
	def __init__(self, coordArray):
		self.direction = coordArray
	def cross(self, other):
		return np.cross(self.direction, other.direction)
	def magnitude(self):
		return np.linalg.norm(self.direction)
	def __str__(self):
		return str(self.direction)

class CoilElement:
	def __init__(self, start, direction, current):
		self.start = start
		self.direction = direction
		self.current = current
	def dB(self, point):
		r = Vector(point.coordinates - self.start.coordinates)
		return self.current*mu_0/(4*pi)*self.direction.cross(r)/(r.magnitude()**3.0)
	def __str__(self):
		return str((str(self.start), str(self.direction)))

class Coil:
	def __init__(self, x_location, current, r, n_chunks):
		points = []
		angle = 2*pi/(n_chunks)
		start_point = Point(np.array((0, r, 0)))
		for i in range(n_chunks):
			points.append(Point(np.array((x_location, r*cos(angle*i), r*sin(angle*i)))))
		self.coilElements = []
		for i in range(len(points)):
			start = points[i]
			end = points[(i+1)%len(points)]
			direction = Vector(end.coordinates-start.coordinates)
			self.coilElements.append(CoilElement(start, direction, current))
	def fieldAt(self, position):
		field = np.array((0.0,0.0,0.0))
		for element in self.coilElements:
			field += element.dB(position) 
		return Vector(field)

	def __str__(self):
		foo = []
		for x in self.coilElements:
			foo.append(str(x))
		return str(foo)

current = 1.0/mu_0
#Core task 1 and task 2 and sup 1, just change values of current
coil1 = Coil(0.5, current, r, 4000)
coil2 = Coil(0.5, -current, r, 4000)


#plotting x 

# xs = np.linspace(-10, 10, 11)
# ys = np.linspace(-10, 10, 11)
# magfield = map(lambda g: (coil.fieldAt(Point(np.array((g,0,0))))).magnitude(), xs)
# theoretical_magfield = map(lambda z: mu_0/(2.0)*current/(z**2+r**2)**(3/2), xs)

# difference = []
# for i in range(len(magfield)):
# 	difference.append(theoretical_magfield[i]- magfield[i])

# fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)


# ax0.plot(xs, magfield, '-')
# ax0.set_title('Calculated')
# ax0.set_xlabel('distance from coil')
# ax0.set_ylabel('Calcualted Field Strength')

# ax1.plot(xs, theoretical_magfield, '-')
# ax1.set_title('Theoretical')
# ax1.set_xlabel('distance from coil')
# ax1.set_ylabel('Calcualted Field Strength')

# ax2.plot(xs, difference, '-')
# ax2.set_title('Difference')
# ax2.set_xlabel('distance from coil')
# ax2.set_ylabel('Difference')
# plt.savefig('xplots')
# plt.show()


#supplementary task 2
N = 2
xpoints = np.linspace(-10*r, 10*r, N)
coils = map(lambda g: Coil(g, current, r, 4000), xpoints)
def sum(coils, point):
	b_field = np.array((0.0,0.0,0.0))
	for c in coils:
		b_field += c.fieldAt(point).direction
	return (b_field[0], b_field[1], b_field[2], Vector(b_field).magnitude())



# import csv 
# f = open('reversecoil.csv', 'w')
# write_points = []
# xs = np.linspace(-0.05, 0.05, 11)
# ys = np.linspace(-0.05, 0.05, 11)

# writer = csv.writer(f, delimiter = '\t')

# for y in ys:
# 	write_points = []
# 	for x in xs:
# 		b_vector = coil1.fieldAt(Point(np.array((x,y, 0.0)))).direction 
# 		b_vector2 = coil2.fieldAt(Point(np.array((x,y,0.0)))).direction

# 		b_field = b_vector + b_vector2
# 		bx = b_field[0]
# 		by = b_field[1]
# 		bz = b_field[2]
# 		bmag = Vector(b_field).magnitude()
# 		write_points.append((x,y,bx,by,bz,bmag))
# 	writer.writerows(write_points)
# 	writer.writerow([])

