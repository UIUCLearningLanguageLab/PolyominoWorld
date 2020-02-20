


param2requests = {
    'hidden_size': [32, 64, 128],
    'learning_rate': [0.01, 0.1],
    'y_type': ['WorldState', 'FeatureVector']
}

param2default = {
    'training_file': 'w6-6_s9_c8_0_100_0.csv',
    'test_file': 'w6-6_s9_c8_0_10_0.csv',
    'network_file': None,
    'learning_rate': 0.20,
    'num_epochs': 10000,
    'weight_init': 0.01,
    'output_freq': 20,
    'verbose': False,
    'x_type': 'WorldState',
    'shuffle_sequences': True,
    'shuffle_events': False,
    'included_features': [1, 1, 1, 0],  # Include': Shape, Size, Color, Action,
    'hidden_size': 32,
    'y_type': 'WorldState',
}

param2debug = {
    'num_epochs': 10,
}