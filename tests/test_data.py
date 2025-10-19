from data import load_countdown

samples = load_countdown("train")
print(samples[0]["prompt"])