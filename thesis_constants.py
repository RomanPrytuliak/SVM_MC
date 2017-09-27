# The script containing the fuzzy patterns to insert at each Monte Carlo test series

from collections import OrderedDict
from ml_classes import EnrichedDimer

MC_patterns = OrderedDict([
    ('A', [EnrichedDimer('acidic', 'aromatic', 0, occur_prob = 1.0)]),
    ('B', [EnrichedDimer('acidic', 'aromatic', 0, occur_prob = 0.5)]),
    ('C', [EnrichedDimer('acidic', 'aromatic', 4, occur_prob = 1.0)]),
    ('D', [EnrichedDimer('polar', 'small', 0, occur_prob = 1.0)]),
    ('E', [EnrichedDimer('polar', 'small', 0, occur_prob = 0.5)]),
    ('F', [EnrichedDimer('ST', 'proline', 0, occur_prob = 1.0)]),
    ('G', []),
    ('H', [EnrichedDimer('polar', 'small', 0, occur_prob = 1.0) for x in range(3)]),
    ('I', [EnrichedDimer('vowels', 'first_five', 2, occur_prob = 1.0)]),
    ('J', [EnrichedDimer('charged', 'aliphatic', 2, occur_prob = 1.0)]),
    ('K', [EnrichedDimer('every_fifth', 'navy', 0, occur_prob = 1.0)]),
    ('L', [EnrichedDimer('vowels', 'first_five', 2, occur_prob = 1.0), EnrichedDimer('every_fifth', 'navy', 0, occur_prob = 1.0)])
])