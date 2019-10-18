from numpy import load

data = load('toronto_face.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])



