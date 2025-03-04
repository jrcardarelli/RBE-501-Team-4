import numpy as np

class FwdKin:
    def __init__(self):
        pass

    def axis_angle_2_rot(self, omega, theta):
        ssm = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        R = np.eye(3) + np.sin(theta) * ssm + (1 - np.cos(theta)) * np.dot(ssm, ssm)
        return R

    def twist2ht(self, S, theta):
        # Skew-symmetric matrix for the rotation part
        ssm = np.array([
            [0, -S[2], S[1]],
            [S[2], 0, -S[0]],
            [-S[1], S[0], 0]
        ])
        # Compute the rotation matrix R
        R = np.eye(3) + np.sin(theta) * ssm + (1 - np.cos(theta)) * np.dot(ssm, ssm)
        # Compute the translation vector P
        P = (np.eye(3) * theta + (1 - np.cos(theta)) * ssm + (theta - np.sin(theta)) * np.dot(ssm, ssm)) @ S[3:]
        # Construct the homogeneous transformation matrix T
        T = np.vstack([
            np.hstack([R, P.reshape(3, 1)]),
            np.array([0, 0, 0, 1])
        ])
        return T
    
    def fkine(self, S, M, q):
        # Initialize the transformation matrix as an identity matrix
        T = np.eye(4)
        
        # Iterate over each column in S
        for i in range(S.shape[1]):
            Si = S[:, i]
            
            # Skew-symmetric matrix for the current column
            ssm = np.array([[0, -Si[2], Si[1]],
                        [Si[2], 0, -Si[0]],
                        [-Si[1], Si[0], 0]])
            
            # Rotation matrix for the current joint
            R = np.eye(3) + np.sin(q[i]) * ssm + (1 - np.cos(q[i])) * np.dot(ssm, ssm)
            
            # Translation vector for the current joint
            P = (np.eye(3) * q[i] + (1 - np.cos(q[i])) * ssm + (q[i] - np.sin(q[i])) * np.dot(ssm, ssm)) @ Si[3:]
            
            # Homogeneous transformation matrix for the current joint
            Ti = np.vstack((np.hstack((R, P.reshape(3, 1))), np.array([0, 0, 0, 1])))
            
            # Update the overall transformation matrix
            T = T @ Ti
        
        # Compute the final forward kinematics
        fk = T @ M
        return fk

    def adjoint(self, V, T):
        # Extract the rotation matrix R and translation vector P from T
        R = T[:3, :3]
        P = T[:3, 3]
        # Skew-symmetric matrix for the translation vector P
        ssmP = np.array([
            [0, -P[2], P[1]],
            [P[2], 0, -P[0]],
            [-P[1], P[0], 0]
        ])
        # Construct the adjoint matrix
        A_top = np.hstack([R, np.zeros((3, 3))])
        A_bottom = np.hstack([ssmP @ R, R])
        A = np.vstack([A_top, A_bottom])
        # Multiply the adjoint matrix with the input vector V
        result = A @ V
        return result

    def jacob0(self, S, q):
        # Initialize the Jacobian matrix J with zeros
        J = np.zeros_like(S)
        # Initialize lists to store intermediate matrices
        M = []
        T = []
        A = []
        
        # Compute the transformation matrices M_i for each joint
        for i in range(S.shape[1]):
            Mi = self.twist2ht(S[:, i], q[i])
            M.append(Mi)
        
        # Initialize T with the first transformation matrix
        T.append(M[0])
        
        # Compute the cumulative transformation matrices T_i
        for j in range(1, len(M)):
            Ti = T[j - 1] @ M[j]
            T.append(Ti)
        
        # Compute the adjoint transformations A_i
        for k in range(len(T)):
            # Ensure we do not exceed the number of columns in S
            if k + 1 < S.shape[1]:
                Ai = self.adjoint(S[:, k + 1], T[k])
                A.append(Ai)
        
        # Fill the Jacobian matrix J
        J[:, 0] = S[:, 0]
        for L in range(len(A)):
            J[:, L + 1] = A[L]
        
        return J


'''
# Test inputs
omega = [0, 0, 1]
theta = np.pi / 2
S = np.array([0, 0, 1, 0, 0, 0])
S2 = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
    [0, 0.3, 0],
    [0, 0, -1]
])
q = [np.random.random() * 2 * np.pi, np.random.random() * 2 * np.pi, np.random.random() * 0.3]
R_home = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]
                   ])
t_home = np.array([0.3, 0, 0.3])
M = np.vstack((np.hstack((R_home, t_home.reshape(3, 1))), [0, 0, 0, 1]))

# Create an instance of InvKin
fwd_kin = FwdKin()

# Test the methods
print(fwd_kin.axis_angle_2_rot(omega, theta))
print('--------------------------')
print(fwd_kin.twist2ht(S, theta))
print('--------------------------')
print(fwd_kin.fkine(S2, M, q))
print('--------------------------')
print(fwd_kin.adjoint(S, np.eye(4)))
print('--------------------------')
print(fwd_kin.jacob0(S2, q))
'''
