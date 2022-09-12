"""
orbit.utils.selections

"""
import numpy as np

from Bio.PDB import Select

from orbit.utils import geometry


class ChainOnly(Select):

    def accept_residue(self, residue):
        return residue.get_id()[0].strip() == ''

    def accept_atom(self, atom):
        return not (
            atom.is_disordered() or
            atom.get_altloc() == "A" or
            atom.get_altloc() == "1"
        )


class LigandOnly(Select):

    def __init__(self, ligand_residue):
        self.lig = ligand_residue

    def accept_residue(self, residue):
        return residue == self.lig


class ModelOnly(Select):

    def __init__(self, model_id):
        self.model_id = model_id

    def accept_model(self, model):
        return model.id == self.model_id


class PocketFromLigand(Select):

    def __init__(
            self, r, ligand,
            keep_ligand=False,
            keep_water=False,
            keep_other_het=False,
            keep_hydrogen=False,
            expand_to_residue=True
    ):
        self.r = r
        self.lig = ligand
        self.keep_lig = keep_ligand
        self.keep_wat = keep_water
        self.keep_het = keep_other_het
        self.keep_hyd = keep_hydrogen
        self.expand = expand_to_residue

    @staticmethod
    def min_dist_residue_to_ligand(residue, ligand):
        la, ra = ligand.get_atoms(), residue.get_atoms()
        lc = np.vstack([a.coord for a in la if not a.element or a.element[0] != 'H'])
        rc = np.vstack([a.coord for a in ra if not a.element or a.element[0] != 'H'])
        dmat = geometry.distance_M_N(lc, rc).min()
        return dmat.min() if np.any(dmat) else 1e10

    @staticmethod
    def min_dist_atom_to_ligand(atom, ligand):
        la, ac = ligand.get_atoms(), np.expand_dims(atom.coord, 0)
        lc = np.vstack([a.coord for a in la if not a.element or a.element[0] != 'H'])
        dmat = geometry.distance_M_N(lc, ac).min()
        return dmat.min() if np.any(dmat) else 1e10

    def accept_residue(self, residue):
        if residue == self.lig:
            return self.keep_lig
        elif residue.resname == 'HOH' or residue.resname == 'WAT':
            return self.keep_wat
        elif not self.keep_het and residue.get_id()[0].strip() != '':
            return False
        else:
            if self.expand is True:
                d = self.min_dist_residue_to_ligand(residue, self.lig)
                return d < self.r and residue.id[2].strip() == ''
            return residue.id[2].strip() == ''

    def accept_atom(self, atom):
        d = 0
        if self.expand is False:
            d = self.min_dist_atom_to_ligand(atom, self.lig)
        if atom.element[0] == 'H':
            return self.keep_hyd and d < self.r
        elif not (
            atom.is_disordered() or
            atom.get_altloc() == "A" or
            atom.get_altloc() == "1"
        ):
            return d < self.r
