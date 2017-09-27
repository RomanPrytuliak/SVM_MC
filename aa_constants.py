# The script containing auxiliary constants describing amino acid background frequencies as well as the predefined amino acid groups for the tests

from collections import OrderedDict

AA_FREQ = {'A': 0.0694, 'C': 0.0214, 'D': 0.0493, 'E': 0.0705, 'F': 0.0366, 'G': 0.0686, 'H': 0.0250, 'I': 0.0438, 'K': 0.0568, 'L': 0.0979, 'M': 0.0203, 'N': 0.0366, 'P': 0.0651, 'Q': 0.0448, 'R': 0.0622, 'S': 0.0815,
           'T': 0.0516, 'V': 0.0602, 'W': 0.0122, 'Y': 0.0260}

AA_GROUPS = OrderedDict([('acidic', 'DE'), ('aliphatic', 'ILV'), ('aromatic', 'FHYW'), ('basic', 'HKR'), ('charged', 'DEHKR'), ('hydrophobic', 'ACFILMPVYW'), ('P_substrates', 'STY'), ('polar', 'DEHKNQRST'),
                         ('small', 'ACDGNPSTV'), ('tiny', 'AGS')])
AA_GROUPS_ = OrderedDict([('acidic', 'DE'), ('aliphatic', 'ILV'), ('aromatic', 'FHYW'), ('basic', 'HKR'), ('charged', 'DEHKR'), ('hydrophobic', 'ACFILMPVYW'), ('P_substrates', 'STY'), ('polar', 'DEHKNQRST'),
                          ('small', 'ACDGNPSTV'), ('tiny', 'AGS'), ('proline', 'P'), ('ST', 'ST'), ('vowels', 'AEIY'), ('first_five', 'ACDEF'), ('every_fifth', 'FLRY'), ('navy', 'ANVY')])