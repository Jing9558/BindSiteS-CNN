"""
orbit.datasets.toughc1

"""
import os
import pickle

from orbit.datasets.dataset import BindingSiteDataset

class ToughC1(BindingSiteDataset):
    """"""

    def __init__(self, engine='fpocket'):
        super(ToughC1, self).__init__('TOUGH-C1')
        assert engine in ('fpocket', 'lpc'), f'Unknown pocket engine: {engine}!'
        self._engine = engine

    @property
    def pocket_engine(self):
        return self._engine

    def get_entries(self, include_metadata=True):
        engine = self.pocket_engine
        entries, sdir = [], self.root
        mapping = os.path.join(self.root, 'pdbcode_mappings.pkl')
        code5_uniprot, code5_seqclust = None, None
        if include_metadata is True and os.path.exists(mapping):
            mapping = pickle.load(open(mapping, 'rb'))
            code5_uniprot = mapping['code5_uniprot']
            code5_seqclust = mapping['code5_seqclust']
        for structure in self.get_structures_list():
            code5 = structure[0][:5]
            code7 = structure[0]
            class_label = structure[1]
            entries.append({
                'code5': code5,
                'code': code5[:4],
                'class_label': class_label,
                'protein': '{0}/protein-{2}/{1}.pdb'.format(sdir, code5, class_label),
                'ligand': '{0}/ligand-{2}/{1}.sdf'.format(sdir, code7, class_label),
                'pocket': '{0}/pocket-{3}-{2}/{1}.pdb'.format(sdir, code7, class_label, engine),
                'uniprot': code5_uniprot[code5] if code5_uniprot else 'None',
                'seqclust': code5_seqclust[code5] if code5_seqclust else 'None',
            })
        return entries

    def get_entry_splits(self, fold_nr, strategy='seqid', n_folds=5, seed=42):
        from sklearn.model_selection import GroupShuffleSplit, KFold
        entries = self.get_entries(include_metadata=True)
        entries = list(filter(lambda entry: entry['class_label'] != 'steroid', entries))
        entries = list(filter(lambda entry: os.path.exists(self.root+'/TOUGH-C1_surfaces_lpc/'+entry['code5']+'.ply') is True, entries))
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

    def get_steroid_entries(self):
        entries = self.get_entries(include_metadata=True)
        entries = list(filter(lambda entry: entry['class_label'] == 'steroid', entries))
        entries = list(filter(lambda entry: os.path.exists(self.root+'/TOUGH-C1_surfaces_lpc/'+entry['code5']+'.ply') is True, entries))
        return entries

    def get_pairs(self,entries):
        pos_pairs = []
        neg_pairs = []
        for i in range(len(entries)):
            code5_i = entries[i]['code5']
            class_label_i = entries[i]['class_label']
            if class_label_i in ['heme','nucleotide']:
                for j in range(i+1, len(entries)):
                    code5_j = entries[j]['code5']
                    class_label_j = entries[j]['class_label']
                    pair = [code5_i, code5_j]
                    if class_label_i == class_label_j:
                        pos_pairs.append(pair)
                    else:
                        neg_pairs.append(pair)
        return pos_pairs, neg_pairs


    def _read_list_file(self, filename):
        with open(os.path.join(self.root, filename)) as file:
            data = [line.strip() for line in file.readlines()]
        return data

    def get_nucleotide_list(self):
        return self._read_list_file('nucleotide.list')

    def get_heme_list(self):
        return self._read_list_file('heme.list')

    def get_steroid_list(self):
        return self._read_list_file('steroid.list')

    def get_control_list(self):
        return self._read_list_file('control.list')

    def get_structures_list(self):
        structures = []
        structures.extend([(x, 'nucleotide') for x in self.get_nucleotide_list()])
        structures.extend([(x, 'heme') for x in self.get_heme_list()])
        structures.extend([(x, 'steroid') for x in self.get_steroid_list()])
        structures.extend([(x, 'control') for x in self.get_control_list()])
        return structures
