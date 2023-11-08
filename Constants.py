class Betas:
    LOW = [{0: 0.7, 1: 0.8, 2: 0.9, 3: 1, 4: 1.1},
              [-90000, -60000, -10000, 0, 2],
              {0: -0.3, 1: -0.2, 2: -0.15, 3: -0.1, 4: -0.05},
              {0: -1.5, 1: -1, 2: -0.5, 3: -0.25, 4: -0.1}]

    MEDIUM = [{0: 0.7, 1: 0.8, 2: 0.9, 3: 1, 4: 1.1},
              [-0.8, -0.55, 0, 0.4, 0.6],
              {0: -0.3, 1: -0.2, 2: -0.15, 3: -0.1, 4: -0.05},
              {0: -2.5, 1: -2, 2: -0.5, 3: -0.25, 4: -0.1}]

    HIGH = [{0: 0.5, 1: 0.6, 2: 1, 3: 1.2, 4: 1.3},
              [-1.85, -0.8, 0, 0.6, 0.8],
              {0: -0.6, 1: -0.4, 2: -0.25, 3: -0.15, 4: -0.1},
              {0: -3, 1: -2.5, 2: -0.75, 3: -0.45, 4: -0.2}]

    VERYHIGH = [{0: -0.4, 1: -0.2, 2: 0.75, 3: 1, 4: 1.1},
              [-2, -1.2, -1, 0.3, 0.5],
              {0: -1, 1: -0.6, 2: -0.5, 3: -0.25, 4: -0.1},
              {0: -3.2, 1: -2.7, 2: -1.75, 3: -0.45, 4: -0.2}]

    EXTREME = [{0: -100, 1: -500, 2: -100, 3: -100, 4: -100},
              [-0.8, -0.55, 0, 0.4, 0.6],
              {0: -0.3, 1: -0.2, 2: -0.15, 3: -0.1, 4: -0.05},
              {0: -2.5, 1: -2, 2: -0.5, 3: -0.25, 4: -0.1}]