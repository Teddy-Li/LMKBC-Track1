"""
Verifies that all answers to ChemicalCompoundElement are indeed chemical elements.
"""

import json


all_elements = ['hydrogen', 'helium', 'lithium', 'beryllium', 'boron', 'carbon', 'nitrogen', 'oxygen', 'fluorine', 'neon',
                    'sodium', 'magnesium', 'aluminium', 'silicon', 'phosphorus', 'sulfur', 'chlorine', 'argon', 'potassium', 'calcium',
                    'scandium', 'titanium', 'vanadium', 'chromium', 'manganese', 'iron', 'cobalt', 'nickel', 'copper', 'zinc',
                    'gallium', 'germanium', 'arsenic', 'selenium', 'bromine', 'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium',
                    'niobium', 'molybdenum', 'technetium', 'ruthenium', 'rhodium', 'palladium', 'silver', 'cadmium', 'indium', 'tin',
                    'antimony', 'tellurium', 'iodine', 'xenon', 'caesium', 'barium', 'lanthanum', 'cerium', 'praseodymium', 'neodymium',
                    'promethium', 'samarium', 'europium', 'gadolinium', 'terbium', 'dysprosium', 'holmium', 'erbium', 'thulium', 'ytterbium',
                    'lutetium', 'hafnium', 'tantalum', 'tungsten', 'rhenium', 'osmium', 'iridium', 'platinum', 'gold', 'mercury',
                    'thallium', 'lead', 'bismuth', 'polonium', 'astatine', 'radon', 'francium', 'radium', 'actinium', 'thorium',
                    'protactinium', 'uranium', 'neptunium', 'plutonium', 'americium', 'curium', 'berkelium', 'californium', 'einsteinium', 'fermium',
                    'mendelevium', 'nobelium', 'lawrencium', 'rutherfordium', 'dubnium', 'seaborgium', 'bohrium', 'hassium', 'meitnerium', 'darmstadtium',
                    'roentgenium', 'copernicium', 'nihonium', 'flerovium', 'moscovium', 'livermorium', 'tennessine', 'oganesson']

for subset in ['train', 'dev']:
    with open(f"./data/{subset}.jsonl", 'r', encoding='utf8') as fp:
        for line in fp:
            item = json.loads(line)
            if item['Relation'] != 'ChemicalCompoundElement':
                continue
            for obj in item['ObjectEntities']:
                for obj_name in obj:
                    assert obj_name in all_elements, print(f"Answer object {obj} not among the elements list!")
