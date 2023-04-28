import numpy as np


def diff_entropy_for_mean_pk(gauss_dim, N_pts, scaler, N, epsilon):
    concave_count = 0
    for i in range(N):
        A = np.random.rand(gauss_dim, N_pts)
        pkA = A[:, 0][:, np.newaxis]  # just take a point a call it our pk
        centered_dataA = A - pkA
        covA = (centered_dataA@(centered_dataA.T)) / N_pts
        entropyA = np.log(scaler*np.linalg.det(covA + epsilon))

        B = np.random.rand(gauss_dim, N_pts)
        pkB = B[:, 0][:, np.newaxis]  # just take a point a call it our pk
        centered_dataB = B - pkB
        covB = (centered_dataB@(centered_dataB.T)) / N_pts
        entropyB = np.log(scaler*np.linalg.det(covB + epsilon))

        H_sep = (entropyA + entropyB)/(2*N_pts)

        centeredAB = np.hstack((A-pkA, B-pkA))
        covAB = (centeredAB@(centeredAB.T))/(2*N_pts)
        H_joint = np.log(scaler*np.linalg.det(covAB + epsilon))

        if H_sep < H_joint:
            concave_count += 1
    return concave_count/N


def diff_entropy_mean_of_points_in_sphere(gauss_dim, N_pts, scaler, N, epsilon):
    concave_count = 0
    for i in range(N):
        A = np.random.rand(gauss_dim, N_pts)
        covA = np.cov(A)*(N_pts-1)/N_pts
        entropyA = np.log(scaler*np.linalg.det(covA+epsilon))
        B = np.random.rand(gauss_dim, N_pts)
        covB = np.cov(B)*(N_pts-1)/N_pts
        entropyB = np.log(scaler*np.linalg.det(covB+epsilon))

        H_sep = (entropyA + entropyB)/(2*N_pts)
        AB = np.hstack((A, B))
        covAB = np.cov(AB)
        H_joint = np.log(scaler*np.linalg.det(covAB+epsilon))

        if H_sep < H_joint:
            concave_count += 1
    return concave_count/N


def main():
    gauss_dim = 3
    scaler = (2*np.pi*np.exp(1))**gauss_dim
    N_pts = 10
    N = 10000
    epsilon = 0

    print(
        f"Concave probability mean of points: {diff_entropy_mean_of_points_in_sphere(gauss_dim, N_pts, scaler, N, epsilon)}")
    print(f"Concave probability mean pk: {diff_entropy_for_mean_pk(gauss_dim, N_pts, scaler, N, epsilon)}")


if __name__ == "__main__":
    main()
