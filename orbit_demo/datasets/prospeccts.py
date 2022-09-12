"""
orbit.datasets.prospeccts

"""
import pickle
import os

from orbit.datasets.dataset import BindingSiteDataset


class ProSPECCTs(BindingSiteDataset):
    """ProSPECCTs dataset by Ehrt et al (http://www.ewit.ccb.tu-dortmund.de/ag-koch/prospeccts/)

    Datasets
    --------
    P1 : Structures with identical sequences
    (sensitivity with respect to binding-site definition)

    P1.2 : Structures with identical sequences and similar ligands
    (impact of ligand diversity on binding site comparison)

    P2 : NMR structures
    (sensitivity with respect to the binding site flexibility)

    P3 : Decoy set 1
    (differentiation between sites with different physiochemical properties)

    P4 : Decoy Set 2
    (differentiation between sites with different physiochemical & shape properties)

    P5 : Kahraman data set (No phosphate)
    (classification of proteins binding to identical ligands and co-factors)

    P5.2 : Kahraman data set
    (classification of proteins binding to identical ligands and co-factors)

    P6 : Barelier data set (No co-factors)
    (identification of distant relationships between binding sites with identical ligands
    which 'observe' similar environments)

    P6.2 : Barelier data set
    (identification of distant relationships between binding sites with identical ligands
    which 'observe' similar environments)

    P7 : Review data set
    (recovery of known binding site similarities within a diverse set of proteins)

    """
    DB_NAMES = {'P1', 'P1.2', 'P2', 'P3', 'P4', 'P5', 'P5.2', 'P6', 'P6.2', 'P7'}

    URL = 'http://www.ewit.ccb.tu-dortmund.de/ag-koch/prospeccts/'

    def __init__(self, db_name):
        super(ProSPECCTs, self).__init__('prospeccts')
        self.db_name = db_name
        assert self.db_name in self.DB_NAMES, f'Unknown database "{db_name}"!'

    @property
    def structure_dir(self):
        dir1, dir2, list_fn = self._get_db_path()
        return os.path.join(self.root, dir1, dir2)

    def get_entries(self, include_metadata=True):
        entries, pdbs, sdir = [], set(), self.structure_dir
        mapping = os.path.join(self.root, 'pdbcode_mappings.pkl')
        for id1, id2, _ in self.get_pairs():
            pdbs.update({id1, id2})
        code5_uniprot, code5_seqclust = None, None
        if include_metadata is True and os.path.exists(mapping):
            mapping = pickle.load(open(mapping, 'rb'))
            code5_uniprot = mapping['code5_uniprot']
            code5_seqclust = mapping['code5_seqclust']
        for code5 in pdbs:
            entries.append({
                'code5': code5,
                'code': code5[:4],
                'uniprot': code5_uniprot[code5] if code5_uniprot else 'None',
                'seqclust': code5_seqclust[code5] if code5_seqclust else 'None',
                'protein': '{0}/{1}.pdb'.format(sdir, code5+'_clean'),
                'ligand': '{0}/{1}.pdb'.format(sdir, code5+'_ligand_1'),
                'pocket': '{0}/{1}.pdb'.format(sdir, code5+'_pocket_1')
            })
        return entries

    def get_pairs(self):
        dir1, _, list_fn = self._get_db_path()
        path = os.path.join(self.root, dir1, list_fn)
        with open(path) as f:
            pairs = [line.strip().split(',') for line in f.readlines()]
        return pairs

    def get_positive_pairs(self):
        return [x[:2] for x in self.get_pairs() if x[2] == 'active']

    def get_negative_pairs(self):
        return [x[:2] for x in self.get_pairs() if x[2] == 'inactive']

    def __repr__(self):
        return '{}(dataset: {}, entries: {})'.format(
            self.__class__.__name__,
            self.db_name,
            len(self)
        )
