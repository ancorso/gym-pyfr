import numpy as np
import math

# Get the normalized singular values from the matrix
def normalized_singular_values(X):
    _, S, _ = np.linalg.svd(X)
    return np.cumsum(S[1:]**2) / np.sum(S[1:]**2)

# Get the number of modes to keep for a specified amount of the energy to be maintained.
# Rounds up to the closest even number
def num_modes(S, retained_energy):
    r = np.argmax(np.cumsum(S[1:]**2) / np.sum(S[1:]**2) > retained_energy)
    if r == 0: 
        r = S[1:].shape[0]

    modes = min(S.shape[0],  int(math.ceil((r+1)/2)*2))
    return modes

# Compute the dynamic mode decomposition with control
def DMDc(O, Xp, retained_energy = 0.99, num_modes_override = None):
    print("retained energy: ", retained_energy)
    n = Xp.shape[0]
    q = O.shape[0] - n

    # Compute the singular value decomposition of Omega (the input space)
    U, S, V = np.linalg.svd(O, full_matrices=False)
    V = V.T
    r = num_modes(S, retained_energy) if num_modes_override is None else num_modes_override

    # Truncate the matrices
    U_til, S_til, V_til = U[:,:r], np.diag(S[:r]), V[:, :r]

    # Compute this efficient SVD of the output space Xp
    U, S, V = np.linalg.svd(Xp, full_matrices=False)
    V = V.T
    rp = num_modes(S, retained_energy) if num_modes_override is None else num_modes_override
    U_hat = U[:,:rp] # note that U_hat' * U_hat \approx I


    U_1 = U_til[:n, :]
    U_2 = U_til[n:n+q, :]
    S_inv = np.linalg.inv(S_til)

    A = U_hat.T @ Xp @ V_til @ S_inv @ U_1.T @ U_hat
    B = U_hat.T @ Xp @ V_til @ S_inv @ U_2.T

    D, W = np.linalg.eig(A)

    new_U_hat = (Xp @ V_til @ S_inv) @ (U_1.T @ U_hat)
    P =  new_U_hat @ W
    # transform = np.linalg.pinv(new_U_hat)
    transform = np.transpose(U_hat)

    return A, B, P, W, transform


