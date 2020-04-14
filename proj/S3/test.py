from textgenrnn import textgenrnn

textgen = textgenrnn(name="new_model")
textgen.train_from_file('jokes2.txt', num_epochs=5)

joke = textgen.generate(return_as_list=True)[0]

print(joke)