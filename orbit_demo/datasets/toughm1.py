"""
orbit.datasets.toughm1

"""
import pickle
import os

from orbit.datasets.dataset import BindingSiteDataset


class ToughM1(BindingSiteDataset):

    def __init__(self):
        super(ToughM1, self).__init__('TOUGH-M1')
        self._n_pos, self._n_neg = None, None

    @property
    def structure_dir(self):
        return os.path.join(self.root, 'TOUGH-M1_dataset')

    @property
    def num_positive_pairs(self):
        if not self._n_pos:
            self._n_pos = len(self.get_positive_pairs())
            return self.num_positive_pairs
        return self._n_pos

    @property
    def num_negative_pairs(self):
        if not self._n_neg:
            self._n_neg = len(self.get_negative_pairs())
            return self.num_negative_pairs
        return self._n_neg

    def get_entries(self, include_metadata=True):
        entries, sdir = [], self.structure_dir
        mapping = os.path.join(self.root, 'pdbcode_mappings.pkl')
        code5_uniprot, code5_seqclust = None, None
        if include_metadata is True and os.path.exists(mapping):
            mapping = pickle.load(open(mapping, 'rb'))
            code5_uniprot = mapping['code5_uniprot']
            code5_seqclust = mapping['code5_seqclust']
        for code5, pocket in self.get_pocket_list():
            entries.append({
                'code5': code5,
                'code': code5[:4],
                'protein': '{0}/{1}/{1}.pdb'.format(sdir, code5),
                'ligand': '{0}/{1}/{1}00.sdf'.format(sdir, code5),
                'pocket': '{0}/{1}/{1}_out/pockets/pocket{2}_atm.pdb'.format(sdir, code5, pocket),
                'uniprot': code5_uniprot[code5] if code5_uniprot else 'None',
                'seqclust': code5_seqclust[code5] if code5_seqclust else 'None',
            })
        return entries

    def get_entry_splits(self, fold_nr, strategy='seqid', n_folds=5, seed=42):
        from sklearn.model_selection import GroupShuffleSplit, KFold
        entries = self.get_entries(include_metadata=True)
        entries = list(filter(lambda entry: os.path.exists(self.root+'/TOUGH-M1_surfaces_ligand/'+entry['code5']+'.ply') is True, entries))
        entries = list(filter(lambda entry: os.path.exists(self.root+'/TOUGH-M1_surfaces_1/'+entry['code5']+'.ply') is True, entries))
        if strategy == 'uniprot':
            splitter = GroupShuffleSplit(n_splits=n_folds, test_size=1.0 / n_folds, random_state=seed)
            entries = list(filter(lambda entry: entry['uniprot'] != 'None', entries))
            folds = list(splitter.split(entries, groups=[i['uniprot'] for i in entries]))
            train_idx, test_idx = folds[fold_nr]
            return [entries[i] for i in train_idx], [entries[i] for i in test_idx]
        elif strategy == 'pdb':
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            folds = list(splitter.split(entries))
            train_idx, test_idx = folds[fold_nr]
            return [entries[i] for i in train_idx], [entries[i] for i in test_idx]
        elif strategy == 'seqid':
            splitter = GroupShuffleSplit(n_splits=n_folds, test_size=1.0 / n_folds, random_state=seed)
            entries = list(filter(lambda entry: entry['seqclust'] != 'None', entries))
            folds = list(splitter.split(entries, groups=[i['seqclust'] for i in entries]))
            train_idx, test_idx = folds[fold_nr]
            return [entries[i] for i in train_idx], [entries[i] for i in test_idx]
        elif strategy is None:
            return entries, entries
        else:
            raise ValueError(f'Unknown split strategy "{strategy}"!')

    def get_target_list(self):
        targets = []
        target_fn = os.path.join(self.root_dir, 'TOUGH-M1_target.list')
        with open(target_fn, 'r') as f:
            for line in f.readlines():
                targets.append(line.strip())
        return targets

    def get_pocket_list(self):
        pockets = []
        pocket_fn = os.path.join(self.root, 'TOUGH-M1_pocket.list')
        with open(pocket_fn, 'r') as f:
            for line in f.readlines():
                code5, pocket_nr, _ = line.split()
                pocket_nr = int(pocket_nr) - 1
                pockets.append((code5, pocket_nr))
        return pockets

    def get_positive_pairs(self):
        with open(os.path.join(self.root, 'TOUGH-M1_positive.list')) as f:
            pos_pairs = [line.split()[:2] for line in f.readlines()]
        return pos_pairs

    def get_negative_pairs(self):
        with open(os.path.join(self.root, 'TOUGH-M1_negative.list')) as f:
            neg_pairs = [line.split()[:2] for line in f.readlines()]
        return neg_pairs