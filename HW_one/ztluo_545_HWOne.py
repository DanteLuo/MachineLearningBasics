import numpy as np
from PIL import Image

im_orignial = Image.open("harvey.jpg").convert("L")
# im_orignial.show()
X = np.asarray(im_orignial)

U, s, V = np.linalg.svd(X,full_matrices=True)

# print(X.shape[0],X.shape[1])

Sigma = np.zeros((U.shape[0],V.shape[1]))
Sigma[:s.shape[0],:s.shape[0]] = np.diag(s)

print(Sigma.shape[0],Sigma.shape[1])

k_index = [2,10,40]

# X_new = np.dot(np.dot(U,Sigma),V)
# im = Image.fromarray(X_new)
# im.show()

for i in k_index:
    u = U[:,0:i]
    sigma = Sigma[0:i,0:i]
    v = V[0:i,:]

    print("Size of u is ",u.shape[0]," and ",u.shape[1]," and size of sigma is ",sigma.shape[0]," and ",sigma.shape[1]," and size of v is ",v.shape[0]," and ",v.shape[1])
    X_new = np.dot(np.dot(u,sigma),v)

    im = Image.fromarray(X_new)
    # im.show()
    # im.save(pic_filename,"PNG")

    F_norm = np.linalg.norm((X-X_new),'fro') / np.linalg.norm(X,'fro')

    print("The F_norm is ",F_norm,"Numbers needed for each k is ",(u.shape[0]*i+i+v.shape[1]*i))

