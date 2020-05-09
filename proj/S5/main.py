
from textgenrnn import textgenrnn


# load data file name and given
file_name = 'jokes.csv'
model_name = 'joke_bot'

model_config = {
    # number of LSTM cells of each layer (128 or 256)
    'rnn_size': 128,
    # number of LSTM layers (2 or greater)
    'rnn_layers': 3,
    # evaluate text both forward and backward
    'rnn_bidirectional': False,
    # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_length': 30,
    # True if you want to train a word-level model instead of a character-level model (requires more data and smaller max_length)
    'word_level': False,
    # (word-level model only) maximum number of words to model; the rest will be ignored
    'max_words': 10000,
}

train_config = {
    # True if each text entry has its own line in the source file
    'line_delimited': True,
    # set higher to train the model for longer
    'num_epochs': 20,
    # generates sample text from model after given number of epochs
    'gen_epochs': 5,
    # proportion of input data to train on:
    'train_size': 1,
    # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'dropout': 0.0,
    # Can't use because we don't have class labels (funny, unfunny, etc.)
    'validation': False,
    # If we are using a CSV file then we set this to true
    'is_csv': True
}

textgen = textgenrnn(name=model_name)

textgen.train_from_file(
    file_path=file_name,
    new_model=True,
    num_epochs=train_config['num_epochs'],
    gen_epochs=train_config['gen_epochs'],
    batch_size=1024,
    train_size=train_config['train_size'],
    dropout=train_config['dropout'],
    validation=train_config['validation'],
    is_csv=train_config['is_csv'],
    rnn_layers=model_config['rnn_layers'],
    rnn_size=model_config['rnn_size'],
    rnn_bidirectional=model_config['rnn_bidirectional'],
    max_length=model_config['max_length'],
    dim_embeddings=100,
    word_level=model_config['word_level'])


# this temperature schedule cycles between 1 very unexpected token, 1 unexpected token, 2 expected tokens, repeat.
# changing the temperature schedule can result in wildly different output!
temperature = [1.0, 0.5, 0.2, 0.2]

# total number of jokes to be generated
n = 1000
# maximum number of tokens in each joke generated
max_gen_length = 60 if model_config['word_level'] else 300

textgen.generate_to_file("output.txt",
                         temperature=temperature,
                         n=n,
                         max_gen_length=max_gen_length)
