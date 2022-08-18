import csv
import glob
import os
import re
import numpy as np
import torch
import torch.utils.data
import trimesh
import logging
import healpy as hp
from orbit.structure import MolecularSurface
from orbit.datasets import ToughM1, ProSPECCTs
import orbit.utils.geometry
import random
import pandas as pd

logging.getLogger('pyembree').disabled = True


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """

    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def make_sgrid(nside, alpha, beta, gamma):
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=True)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=np.float32)  # shape 3 x npix
    R = rotmat(alpha, beta, gamma, hom_coord=False)
    sgrid = np.einsum('ij,nj->ni', R, coords)  # inner(A,B).T
    return sgrid


def render_model(mesh, sgrid):
    r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
    mesh.trimesh.apply_scale(1 / r)
    r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
    mesh.trimesh.apply_scale(0.99 / r)

    # Cast rays
    # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
    index_tri, index_ray, loc = mesh.trimesh.ray.intersects_id(
        ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3))  # fix bug if loc is empty

    # Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
    grid_hits = sgrid[index_ray]
    grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)

    # Compute the distance from the grid points to the intersection pionts
    dist = np.linalg.norm(grid_hits - loc, axis=-1)
    dist = np.nan_to_num(dist)

    # Compute the distance from origin to the intersection points
    dist_origin = np.linalg.norm(grid_hits, axis=-1)
    dist_origin *= (r / 0.99)
    dist_origin -= 7.46
    dist_origin /= 2.93
    dist_origin = np.nan_to_num(dist_origin)
    dist_origin_im = np.ones(sgrid.shape[0])
    dist_origin_im[index_ray] = dist_origin

    # For each intersection, look up the normal of the triangle that was hit
    normals = mesh.trimesh.face_normals[index_tri]
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Construct spherical images
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    # dist_im = dist_im.reshape(theta.shape)

    # shaded_im = np.zeros(sgrid.shape[0])
    # shaded_im[index_ray] = normals.dot(light_dir)
    # shaded_im = shaded_im.reshape(theta.shape) + 0.4

    n_dot_ray_im = np.zeros(sgrid.shape[0])
    # n_dot_ray_im[index_ray] = np.abs(np.einsum("ij,ij->i", normals, grid_hits_normalized))
    n_dot_ray_im[index_ray] = np.einsum("ij,ij->i", normalized_normals, grid_hits_normalized)
    n_dot_ray_im[index_ray] = np.nan_to_num(n_dot_ray_im[index_ray])

    nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], normalized_normals[:, 2]
    gx, gy, gz = grid_hits_normalized[:, 0], grid_hits_normalized[:, 1], grid_hits_normalized[:, 2]
    wedge_norm = np.sqrt((nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2 + (ny * gz - nz * gy) ** 2)
    wedge_norm = np.nan_to_num(wedge_norm)
    n_wedge_ray_im = np.zeros(sgrid.shape[0])
    n_wedge_ray_im[index_ray] = wedge_norm

    faces = mesh.faces[index_tri]

    # get atomic_hydrophobicity
    atomic_hydrophobicity = mesh.vertex_properties['atomic_hydrophobicity_4.5'][faces].mean(axis=1)
    atomic_hydrophobicity = np.nan_to_num(atomic_hydrophobicity)
    atomic_hydrophobicity_im = np.zeros(sgrid.shape[0])
    atomic_hydrophobicity_im[index_ray] = atomic_hydrophobicity

    # get acc map
    ACC_map = mesh.vertex_properties['ACC_map'][faces].mean(axis=1)
    ACC_map_im = np.zeros(sgrid.shape[0])
    ACC_map_im[index_ray] = ACC_map

    # get DON map
    DON_map = mesh.vertex_properties['DON_map'][faces].mean(axis=1)
    DON_map_im = np.zeros(sgrid.shape[0])
    DON_map_im[index_ray] = DON_map

    # get ALI map
    ALI_map = mesh.vertex_properties['ALI_map'][faces].mean(axis=1)
    ALI_map_im = np.zeros(sgrid.shape[0])
    ALI_map_im[index_ray] = ALI_map

    # get ARO map
    ARO_map = mesh.vertex_properties['ARO_map'][faces].mean(axis=1)
    ARO_map_im = np.zeros(sgrid.shape[0])
    ARO_map_im[index_ray] = ARO_map

    # gaussian_curvature
    gaussian_curvature = mesh.vertex_properties['gaussian_curvature'][faces].mean(axis=1)
    gaussian_curvature_im = np.zeros(sgrid.shape[0])
    gaussian_curvature_im[index_ray] = gaussian_curvature

    # shape_index
    shape_index = mesh.vertex_properties['shape_index'][faces].mean(axis=1)
    shape_index_im = np.zeros(sgrid.shape[0])
    shape_index_im[index_ray] = shape_index

    # mean_curvature
    mean_curvature = mesh.vertex_properties['mean_curvature'][faces].mean(axis=1)
    mean_curvature_im = np.zeros(sgrid.shape[0])
    mean_curvature_im[index_ray] = mean_curvature

    # get charges value at each vertex in face
    charges = mesh.vertex_properties['potential'][faces].mean(axis=1)
    charges_ = np.copy(charges)
    np.clip(charges_, -5, 5, charges_)
    charges_ -= -5
    charges_ /= 10
    charges = 2 * charges_ - 1
    charges = np.nan_to_num(charges)
    charges_im = np.zeros(sgrid.shape[0])
    charges_im[index_ray] = charges

    # Combine channels to construct final image
    # im = dist_im.reshape((1,) + dist_im.shape)
    im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im, charges_im, atomic_hydrophobicity_im, ACC_map_im, DON_map_im,
                   ALI_map_im, ARO_map_im), axis=0)

    return im


class ToMesh:
    def __init__(self, random_rotations=False, random_translation=0):
        self.rot = random_rotations
        self.tr = random_translation

    def __call__(self, path):
        mesh = MolecularSurface.load_ply(path)

        mesh.apply_translation(-mesh.trimesh.centroid)

        if self.tr > 0:
            tr = np.random.rand() * self.tr
            rot = orbit.utils.geometry.random_rotmat_zyz()
            mesh.apply_transformation(rot)
            mesh.apply_translation([tr, 0, 0])

            if not self.rot:
                mesh.apply_transformation(rot.T)

        if self.rot:
            mesh.apply_transformation(orbit.utils.geometry.random_rotmat_zyz())

        return mesh

    def __repr__(self):
        return self.__class__.__name__ + '(rotation={0}, translation={1})'.format(self.rot, self.tr)


class ProjectOnSphere:
    def __init__(self, nside):
        self.nside = nside
        self.sgrid = make_sgrid(self.nside, alpha=0, beta=0, gamma=0)

    def __call__(self, mesh):
        im = render_model(mesh, self.sgrid)
        npix = self.sgrid.shape[0]
        im = im.reshape(9, npix)

        assert len(im) == 9

        im = im.astype(np.float32).T

        return im

    def __repr__(self):
        return self.__class__.__name__ + '(nside={0})'.format(self.nside)


class CacheNPY:
    def __init__(self, prefix, repeat, transform, pick_randomly=True):
        self.transform = transform
        self.prefix = prefix
        self.repeat = repeat
        self.pick_randomly = pick_randomly

    def check_trans(self, file_path):
        print("transform {}...".format(file_path))
        try:
            return self.transform(file_path)
        except:
            print("Exception during transform of {}".format(file_path))
            raise

    def __call__(self, file_path):
        head, tail = os.path.split(file_path)
        root, _ = os.path.splitext(tail)
        npy_path = os.path.join(head, self.prefix + root + '_{0}.npy')

        exists = [os.path.exists(npy_path.format(i)) for i in range(self.repeat)]

        if self.pick_randomly and all(exists):
            i = np.random.randint(self.repeat)
            try:
                return np.load(npy_path.format(i), allow_pickle=True)
            except OSError:
                exists[i] = False

        if self.pick_randomly:
            img = self.check_trans(file_path)
            np.save(npy_path.format(exists.index(False)), img)

            return img

        output = []
        for i in range(self.repeat):
            try:
                img = np.load(npy_path.format(i))
            except (OSError, FileNotFoundError):
                img = self.check_trans(file_path)
                np.save(npy_path.format(i), img)
            output.append(img)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(prefix={0}, transform={1})'.format(self.prefix, self.transform)


class TOUGHM1Pair(torch.utils.data.Dataset):
    '''
    TOUGHM1Pair
    '''

    def __init__(self, root, dataset, fold_n, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.dir = os.path.join(self.root, 'TOUGH-M1_surfaces_ligand')
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        TM1 = ToughM1()

        train_1, test_1 = TM1.get_entry_splits(fold_n)

        if dataset == 'train':
            self.files = []
            self._code5_map = []
            self.files_dic = {}
            for entry in train_1:
                self.files.append(os.path.join(self.dir, entry['code5'] + '.ply'))
                self._code5_map.append(entry['code5'])
                self.files_dic[entry['code5']] = (os.path.join(self.dir, entry['code5'] + '.ply'))
            filter_fn = lambda p: p[0] in self._code5_map and p[1] in self._code5_map
            positive_pairs = TM1.get_positive_pairs()
            negative_pairs = TM1.get_negative_pairs()
            self._pos_pairs = list(filter(filter_fn, positive_pairs))
            self._neg_pairs = list(filter(filter_fn, negative_pairs))

        elif dataset == 'test':
            self.files = []
            self._code5_map = []
            self.files_dic = {}
            for entry in test_1:
                self.files.append(os.path.join(self.dir, entry['code5'] + '.ply'))
                self._code5_map.append(entry['code5'])
                self.files_dic[entry['code5']] = (os.path.join(self.dir, entry['code5'] + '.ply'))
            filter_fn = lambda p: p[0] in self._code5_map and p[1] in self._code5_map
            positive_pairs = TM1.get_positive_pairs()
            negative_pairs = TM1.get_negative_pairs()
            self._pos_pairs = list(filter(filter_fn, positive_pairs))
            self._neg_pairs = list(filter(filter_fn, negative_pairs))

    def __getitem__(self, index):

        if index % 2 == 0:
            pair = self._pos_pairs[index // 2]
            target = 1

        else:
            pair = random.choice(self._neg_pairs)
            target = 0

        img_i = self.files_dic[pair[0]]
        img_j = self.files_dic[pair[1]]

        if self.transform is not None:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_i, img_j, target

    def __len__(self):
        return 2 * len(self._pos_pairs)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*.ply"))

        return len(files) > 0


class TOUGHM1Pair_test(torch.utils.data.Dataset):
    '''
    TOUGHM1Pair_test
    '''

    def __init__(self, root, dataset, fold_n, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.dir = os.path.join(self.root, 'TOUGH-M1_surfaces_ligand')
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        TM1 = ToughM1()

        train_1, test_1 = TM1.get_entry_splits(fold_n)

        if dataset == 'train':
            self.files = []
            self._code5_map = []
            self.files_dic = {}
            for entry in train_1:
                self.files.append(os.path.join(self.dir, entry['code5'] + '.ply'))
                self._code5_map.append(entry['code5'])
                self.files_dic[entry['code5']] = (os.path.join(self.dir, entry['code5'] + '.ply'))
            filter_fn = lambda p: p[0] in self._code5_map and p[1] in self._code5_map
            positive_pairs = TM1.get_positive_pairs()
            negative_pairs = TM1.get_negative_pairs()
            self._pos_pairs = list(filter(filter_fn, positive_pairs))
            self._neg_pairs = list(filter(filter_fn, negative_pairs))
            self._pairs = self._pos_pairs + self._neg_pairs

        elif dataset == 'test':
            self.files = []
            self._code5_map = []
            self.files_dic = {}
            for entry in test_1:
                self.files.append(os.path.join(self.dir, entry['code5'] + '.ply'))
                self._code5_map.append(entry['code5'])
                self.files_dic[entry['code5']] = (os.path.join(self.dir, entry['code5'] + '.ply'))
            filter_fn = lambda p: p[0] in self._code5_map and p[1] in self._code5_map
            positive_pairs = TM1.get_positive_pairs()
            negative_pairs = TM1.get_negative_pairs()
            self._pos_pairs = list(filter(filter_fn, positive_pairs))
            self._neg_pairs = list(filter(filter_fn, negative_pairs))
            self._pairs = self._pos_pairs + self._neg_pairs

    def __getitem__(self, index):

        if index in range(len(self._pos_pairs)):
            pair = self._pairs[index]
            target = 1

        else:
            pair = self._pairs[index]
            target = 0

        img_i = self.files_dic[pair[0]]
        img_j = self.files_dic[pair[1]]

        if self.transform is not None:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_i, img_j, target

    def __len__(self):
        return len(self._pairs)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*.ply"))

        return len(files) > 0


class ProSPECCTsPairs(torch.utils.data.Dataset):
    '''
    ProSPECCTsPairs
    '''

    def __init__(self, root, db_name, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.dir = os.path.join(self.root, db_name + '_surfaces')
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        PS = ProSPECCTs(db_name)

        entries = PS.get_entries()
        entries = list(
            filter(lambda entry: os.path.exists(os.path.join(self.dir, entry['code5'] + '.ply')) is True, entries))

        self.files = []
        self._code5_map = []
        self.files_dic = {}
        for entry in entries:
            self.files.append(os.path.join(self.dir, entry['code5'] + '.ply'))
            self._code5_map.append(entry['code5'])
            self.files_dic[entry['code5']] = (os.path.join(self.dir, entry['code5'] + '.ply'))
        filter_fn = lambda p: p[0] in self._code5_map and p[1] in self._code5_map
        positive_pairs = PS.get_positive_pairs()
        negative_pairs = PS.get_negative_pairs()
        self._pos_pairs = list(filter(filter_fn, positive_pairs))
        self._neg_pairs = list(filter(filter_fn, negative_pairs))
        self._pairs = self._pos_pairs + self._neg_pairs

    def __getitem__(self, index):

        if index in range(len(self._pos_pairs)):
            pair = self._pairs[index]
            target = 1

        else:
            pair = self._pairs[index]
            target = 0

        img_i = self.files_dic[pair[0]]
        img_j = self.files_dic[pair[1]]

        if self.transform is not None:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_i, img_j, target

    def __len__(self):
        return len(self._pairs)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*.ply"))

        return len(files) > 0