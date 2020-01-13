from cvxpy import quad_form, Variable, Problem, Minimize, norm
import numpy as np

# Perform MPC optimization to find next input
# Following example from CVXPY documentation
def mpc_input(A, B, x0, T, R, u_max):
    state_dim, action_dim = B.shape

    # Define variables
    x = Variable(shape = (state_dim, T+1))
    u = Variable(shape = (action_dim, T))

    # Define costs for states and inputs
    Q = np.eye(state_dim-1)
    R = R*np.eye(action_dim)

    # Construct and solve optimization problem
    cost = 0
    constr = []
    B = np.squeeze(B)

    for t in range(T):
        cost += quad_form(x[1:,t], Q) + quad_form(u[:,t], R)
        constr += [x[:,t+1] == A*x[:,t] + B*u[:,t], norm(u[:,t], 'inf') <= u_max]

    # Sum problem objectives and concatenate constraints
    constr += [x[:,0] == x0]
    prob = Problem(Minimize(cost),constr)
    prob.solve()

    try:
        return u.value[:, 0] # Change if not scalar input
    except:
        print("Error: u was not a scalar")
        return 0.0