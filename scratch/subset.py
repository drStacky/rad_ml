import json


pth = '../experiment/shanghai_build.json'

with open(pth) as f:
    data = json.load(f)

train = [d for d in data if d['kind'] == 'train']
valid = [d for d in data if d['kind'] == 'valid']
test = [d for d in data if d['kind'] == 'test']

print(len(train))
print(len(valid))
print(len(test))

new_train = train[:100]
new_valid = valid[:100]
new_test = test[:100]

new_data = new_train + new_valid + new_test

opth = '../experiment/subset_shanghai_build.json'
with open(opth, 'w') as dst:
    json.dump(new_data, dst, indent=3)

