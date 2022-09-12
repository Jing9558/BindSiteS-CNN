"""
oribit.structure

"""
import numpy as np
import weakref

from orbit_demo.utils import mesh

class MolecularSurface(object):

    def __init__(self, trimesh, parent=None):
        self._parent = weakref.proxy(parent) if parent else None
        self.trimesh = trimesh

    @property
    def parent(self):
        return self._parent

    @property
    def vertices(self):
        return self.trimesh.vertices

    @property
    def faces(self):
        return self.trimesh.faces

    @property
    def vertex_properties(self):
        return self.trimesh.vertex_attributes

    @property
    def kdtree(self):
        return self.trimesh.kdtree

    def is_genus_zero(self):
        return self.trimesh.euler_number == 2

    def apply_transformation(self, matrix):
        self.trimesh.apply_transform(matrix)

    def apply_translation(self, translation):
        self.trimesh.apply_translation(translation)

    def colour_by_vertex_attribute(self, attribute, low=None, high=None, cmap='BuPu'):
        values = self[attribute]
        interp = mesh.visual.interpolate(values, cmap, low, high)
        self.trimesh.visual.vertex_colors = interp

    def shape_signature(self, **kwargs):
        from orbit_demo.utils.feature.shape_signature import ShapeSignatureCalculator
        # Interpolate property from face
        # TODO: maybe add shape signature here.
        if not self.is_genus_zero():
            raise ValueError('Expected a genus-0 surface!')
        calc = ShapeSignatureCalculator(**kwargs)

    def show(self):
        attr = self.trimesh.vertex_attributes
        self.trimesh.vertex_attributes = {}
        vis = self.trimesh.show()
        self.trimesh.vertex_attributes = attr
        return vis

    def save_ply(self, output_file):
        from trimesh.exchange.ply import export_ply
        with open(output_file, 'wb') as ply:
            out = export_ply(self.trimesh, include_attributes=True)
            ply.write(out)

    @classmethod
    def load_ply(cls, input_file, parent=None):
        from trimesh import load_mesh
        ignore = {'x', 'y', 'z', 'red', 'green', 'blue', 'alpha'}
        tri = load_mesh(
            input_file, file_type='ply', fix_texture=False,
            process=False, validate=False
        )
        v = tri.metadata.pop('ply_raw')['vertex']
        for key in v['properties'].keys():
            if key in ignore:
                continue
            tri.vertex_attributes[key] = v['data'][key]
        return cls(tri, parent=parent)

    def __getitem__(self, item):
        return self.trimesh.vertex_attributes[item]

    def __setitem__(self, key, value):
        value = np.asanyarray(value)
        if value.shape[0] != self.trimesh.vertices.shape[0]:
            raise ValueError('Vertex attribute must be the same size as vertices')
        self.trimesh.vertex_attributes[key] = value

    def __repr__(self):
        return '{}(parent={})'.format(
            self.__class__.__name__,
            self.parent
        )
