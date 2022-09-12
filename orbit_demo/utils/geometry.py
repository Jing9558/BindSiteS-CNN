"""
orbit.utils.geometry

"""
import numpy as np
import numba as nb

from scipy.spatial.distance import cdist
from itertools import groupby


class Sphere(object):

    _PDB_FORMAT_STRING = "ATOM{:>7}  C   SPH{:>6}{:>12.3f}{:>8.3f}{:>8.3f}  1.00  {:.2f}\n"

    def __init__(self, center, radius, cluster=0):
        self.center = np.asanyarray(center, dtype=np.float32)
        self.radius = float(radius)
        self.cluster = int(cluster)

    def contains_point(self, point):
        diff = self.center - point
        dist = np.sum(np.square(diff))
        return dist < np.square(self.radius)

    def intersects_other(self, sphere_other):
        dist = distance(self.center, sphere_other.center)
        rsum = self.radius + sphere_other.radius
        return rsum >= dist

    def pdb_line_format(self, atom_number=1):
        args = (atom_number, self.cluster, *self.center, self.radius)
        return self._PDB_FORMAT_STRING.format(*args)

    @staticmethod
    def write_spheres_pdb(spheres, outfile):
        with open(outfile, 'w') as pdb:
            for index, sphere in enumerate(spheres):
                pdb.write(sphere.pdb_line_format(index + 1))

    @classmethod
    def from_pdb_record(cls, record):
        tokens = record.strip().split()
        center = np.array(tokens[5:8], dtype=float)
        radius = float(tokens[9])
        cluster = int(tokens[4])
        return cls(center, radius, cluster)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if (
                self.center == other.center and
                self.radius == other.radius and
                self.cluster == other.cluster
            ):
                return True
        return False

    def __repr__(self):
        return '{_cls}(center={center}, radius={radius})'.format(
            _cls=self.__class__.__name__,
            center=self.center,
            radius=self.radius
        )


def read_pdb_spheres(filepath, seperate_clusters=True):
    spheres = []
    with open(filepath) as sphfile:
        lines = sphfile.readlines()
    for line in lines:
        if 'ATOM' not in line:
            continue
        sphere = Sphere.from_pdb_record(line)
        spheres.append(sphere)
    if seperate_clusters:
        group = groupby(spheres, key=lambda x: x.cluster)
        spheres = [list(g) for _, g in group]
    return spheres


def Rx(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0, 0, 1]
    ])


def Ry(theta):
    return np.array([
        [np.cos(theta),  0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])


def Rz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def rotmat_zyz(psi, theta, phi, homogenous_coord=False):
    r = Rz(psi).dot(Ry(theta)).dot(Rz(phi))
    if homogenous_coord:
        return r
    return r[:3, :3]


def random_rotmat_zyz():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    return rotmat_zyz(a, np.arccos(z), c, True)


def invert_rotmat(rotmat):
    return np.linalg.inv(rotmat)


@nb.jit(nopython=True, cache=True)
def align_vector_to_vector(v1, v2):
    a = np.ascontiguousarray(unit_vector(v1))
    b = np.ascontiguousarray(unit_vector(v2))
    axis = unit_vector(np.cross(a, b))
    angle = np.arccos(np.minimum(np.dot(a, b), np.array([1])))
    if not np.any(axis):  # if collinear
        c = np.zeros((3,))
        c[np.argmin(np.abs(a))] = 0
        axis = unit_vector(np.cross(a, c))
    return axis, angle[0]


@nb.jit(nopython=True, cache=True)
def rotmat_from_axis_angle(axis, angle, homogenous_coord=False):
    s = np.sin(angle)
    c = np.cos(angle)
    t = 1 - c
    x, y, z = unit_vector(axis)
    m = np.array([
        [t*x*x+c, t*x*y-s*z, t*x*z+s*y, 0],
        [t*x*y+s*z, t*y*y+c, t*y*z-s*x, 0],
        [t*x*z-s*y, t*y*z+s*x, t*z*z+c, 0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    if homogenous_coord:
        return m
    return m[:3, :3]


def transform_points(points, matrix, translate=True):
    matrix = np.asanyarray(matrix, order='C', dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)
    if matrix.shape != (4, 4):  # homogenous transformation matrix
        raise ValueError('Transformation matrix must be (4, 4)!')
    if points.shape[0] == 0:
        return points.copy()
    if len(points.shape) != 2 or points.shape[1] + 1 != matrix.shape[1]:
        raise ValueError('matrix shape ({}) doesn\'t match points ({})'.format(
            matrix.shape, points.shape))
    identity = np.abs(matrix - np.eye(matrix.shape[0])).max()
    if identity < 1e-8:
        return np.ascontiguousarray(points.copy())
    dimension = points.shape[1]
    column = np.zeros(len(points)) + int(bool(translate))
    stacked = np.column_stack((points, column))
    transformed = np.dot(matrix, stacked.T).T[:, :dimension]
    transformed = np.ascontiguousarray(transformed)
    return transformed


def transform_around(matrix, point):
    point = np.asanyarray(point)
    matrix = np.asanyarray(matrix)
    dim = len(point)
    if matrix.shape != (dim + 1, dim + 1):
        raise ValueError('matrix must be (d+1, d+1)')
    translate = np.eye(dim + 1)
    translate[:dim, dim] = -point
    result = np.dot(matrix, translate)
    translate[:dim, dim] = point
    result = np.dot(translate, result)
    return result


@nb.jit(nopython=True, cache=True)
def _clamp(val, a, b):
    if val < a:
        return a
    if val > b:
        return b
    return val


@nb.jit(nopython=True, cache=True)
def distance(x, y):
    return np.linalg.norm(x - y)


@nb.jit(nopython=True, cache=True)
def unit_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


@nb.jit(nopython=True, cache=True)
def angle_between_vectors(v1, v2):
    uv1 = unit_vector(v1)
    uv2 = unit_vector(v2)
    dot = np.dot(uv1, uv2)
    angle = np.arccos(_clamp(dot, -1, 1))
    return angle


@nb.jit(nopython=True, cache=True)
def reflection_vector(direction, normal):
    n = normal / np.linalg.norm(normal)
    return direction - (2 * (direction.dot(n)) * n)


@nb.jit(nopython=True, cache=True)
def three_point_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    return angle_between_vectors(v1, v2)


@nb.jit(nopython=True, cache=True)
def dihedral_angle(p1, p2, p3, p4):
    ab = p1 - p2
    cb = p3 - p2
    db = p4 - p3
    u = np.cross(ab, cb)
    v = np.cross(db, cb)
    w = np.cross(u, v)
    angle_uv = angle_between_vectors(u, v)
    try:
        if angle_between_vectors(cb, w) > 0.001:
            angle_uv = -angle_uv
    except Exception:
        pass
    return angle_uv


def bisect(v1, v2):
    uv1 = unit_vector(v1)
    uv2 = unit_vector(v2)
    return uv1 + uv2


@nb.jit(nopython=True, cache=True)
def _cart2sph(coords):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]
    out = np.empty(x.shape + (2,))
    out[..., 0] = np.arccos(z)  # beta
    out[..., 1] = np.arctan2(y, x)  # alpha
    return out


@nb.jit(nopython=True, cache=True)
def _sph2cart(coords):
    beta = coords[..., 0]
    alpha = coords[..., 1]
    r = 1.
    out = np.empty(beta.shape + (3,))
    ct = np.cos(beta)
    cp = np.cos(alpha)
    st = np.sin(beta)
    sp = np.sin(alpha)
    out[..., 0] = r * st * cp  # x
    out[..., 1] = r * st * sp  # y
    out[..., 2] = r * ct       # z
    return out


def change_coordinates(coords, p_from='C', p_to='S'):
    if p_from == p_to:
        return coords
    elif p_from == 'C' and p_to == 'S':
        return _cart2sph(coords)
    elif p_from == 'S' and p_to == 'C':
        return _sph2cart(coords)
    else:
        raise ValueError(f'Unknown conversion: {p_from} -> {p_to}')


def fit_plane_svd(points):
    C = points.mean(axis=0)
    CX = points - C
    U, S, V = np.linalg.svd(CX)
    N = V[-1]  # last right singular vector
    return C, N


@nb.njit(cache=True, nogil=True)
def calc_normal(n, c, o):
    i, j = o - c, n - c
    cross = np.cross(i, j)
    return cross / np.linalg.norm(cross)


@nb.jit(nopython=True, cache=True)
def _point_in_hull(point, equations, eps=1e-12):
    for i in range(equations.shape[0]):
        eq = equations[i]
        if np.dot(eq[:-1], point) + eq[-1] >= eps:
            return 0
    return 1


@nb.jit(nopython=True, parallel=True, cache=True)
def _points_in_hull(points, equations, eps=1e-12):
    len_points = points.shape[0]
    result = np.zeros(len_points, dtype=np.bool8)
    for i in nb.prange(len_points):
        result[i] = _point_in_hull(points[i], equations, eps)
    return result


def points_in_hull(points, hull, eps=1e-12):
    equations = hull.equations
    points = np.asanyarray(points, dtype=np.float64)
    return _points_in_hull(points, equations, eps)


def hull_centroid(hull):
    """Approximate the centroid of the convex hull using the divergence theorem"""
    A = hull.points[hull.simplices[:, 0], :]
    B = hull.points[hull.simplices[:, 1], :]
    C = hull.points[hull.simplices[:, 2], :]
    N = np.cross(B - A, C - A)
    M = np.mean(hull.points[hull.vertices, :], axis=0)
    sign = np.sign(np.sum((A - M) * N, axis=1, keepdims=True))
    N = N * sign
    vol = np.sum(N * A) / 6
    return 1/(2*vol)*(1/24 * np.sum(N*((A+B)**2 + (B+C)**2 + (C+A)**2), axis=0))


def max_diameter(points):
    return cdist(points, points).max()


def distance_M_N(M, N):
    return cdist(M, N)
