import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

digits = load_digits()

x = digits.data
y = digits.target

X = StandardScaler().fit_transform(x)

tsne = TSNE(n_components=2,perplexity=30,random_state=42)

x_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10,7))

scatter= plt.scatter(
    x_tsne[:,0],
    x_tsne[:,1],
    c=y,
    cmap="tab10",
    s=15
)
plt.colorbar(scatter,label="Digit")

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualization of Handwritten Digits")

plt.show()
