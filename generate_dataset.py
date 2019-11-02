import pandas as pd
import numpy as np


np.random.seed(42)


class_distribution = pd.read_csv('class_distribution.csv')
class_distribution = class_distribution.set_index('race')

race_distribution = pd.read_csv('race_distribution.csv')
race_distribution = race_distribution.set_index('race').iloc[:, 0]

races = {
    'dwarf': {'height': ((4, 2), (4, 8)), 'weight': (134, 226), 'age': (300, 50),
              'strength': 0, 'charisma': 0, 'constitution': 2},
    'gnome': {'height': ((3, 1), (3, 7)), 'weight': (37, 43), 'age': (425, 75),
              'strength': 0, 'charisma': 0, 'constitution': 0},
    'elf': {'height': ((4, 8), (6, 2)), 'weight': (92, 170), 'age': (700, 50),
            'strength': 0, 'charisma': 0, 'constitution': 0},
    'halfling': {'height': ((2, 9), (3, 3)), 'weight': (33, 38), 'age': (150, 50),
                 'strength': 0, 'charisma': 0, 'constitution': 0},
    'half-elf': {'height': ((4, 11), (6, 1)), 'weight': (114, 238), 'age': (150, 10),
                 'strength': 1, 'charisma': 2, 'constitution': 1},
    'human': {'height': ((4, 9), (5, 7)), 'weight': (114, 270), 'age': (70, 5),
              'strength': 1, 'charisma': 1, 'constitution': 1},
    'dragonborn': {'height': ((5, 8), (6, 1)), 'weight': (179, 367), 'age': (60, 10),
                   'strength': 2, 'charisma': 1, 'constitution': 0},
    'half-orc': {'height': ((5, 0), (6, 6)), 'weight': (144, 380), 'age': (70, 5),
                 'strength': 2, 'charisma': 0, 'constitution': 1},
    'tiefling': {'height': ((4, 11), (6, 1)), 'weight': (114, 238), 'age': (80, 10),
                 'strength': 0, 'charisma': 2, 'constitution': 0},
}


N = 100_000
characters = []

for race in races:
    n = int(N * race_distribution.loc[race])
    low_height, high_height = races[race]['height']
    low_height = low_height[0] * 13 + low_height[1]
    high_height = high_height[0] * 13 + high_height[1]
    d = (high_height - low_height) / 2
    u = d + low_height
    heights = np.random.normal(u, d*.3, size=n)

    low_weight, high_weight = races[race]['weight']
    d = (high_weight - low_weight) / 2
    u = d + low_weight
    weights = np.random.normal(u, d*.3, size=n)

    age_u, age_v = races[race]['age']
    ages = np.random.normal(age_u, age_v*.3, size=n)

    classes = class_distribution.loc[race]
    class_ = classes.sample(n, weights=classes, replace=True).index

    df = pd.DataFrame({
        'height': heights,
        'weight': weights,
        'age': ages,
        'class': class_,
        'race': [race] * n,
        'strength': [races[race]['strength']] * n,
        'charisma': [races[race]['charisma']] * n,
        'constitution': [races[race]['constitution']] * n,
    })
    characters.append(df)

all_data = pd.concat(characters)

mapper = {
    'dragonborn': 0,
    'dwarf': 0,
    'elf': 1,
    'gnome': 0,
    'half-elf': 1,
    'half-orc': 0,
    'halfling': 0,
    'human': 1,
    'tiefling': 0
}
all_data['most_common_race'] = all_data.race.map(mapper)

all_data.to_csv('characters.csv', index=False)
