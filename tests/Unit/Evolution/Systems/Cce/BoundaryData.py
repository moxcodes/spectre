# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import math


def compute_cartesian_to_angular_coordinates(cos_phi, cos_theta, sin_phi,
                                             sin_theta, extraction_radius):
    return np.array([
        extraction_radius * cos_phi * sin_theta,
        extraction_radius * sin_phi * sin_theta, extraction_radius * cos_theta
    ])


def compute_cartesian_to_angular_jacobian(cos_phi, cos_theta, sin_phi,
                                          sin_theta, extraction_radius):
    return np.array(
        [[cos_phi * sin_theta, sin_phi * sin_theta, cos_theta],
         [
             extraction_radius * cos_phi * cos_theta,
             extraction_radius * sin_phi * cos_theta,
             -extraction_radius * sin_theta
         ], [-extraction_radius * sin_phi, extraction_radius * cos_phi, 0.0]])


#TODO find a different auto-formatter that does better
def compute_cartesian_to_angular_inverse_jacobian(
        cos_phi, cos_theta, sin_phi, sin_theta, extraction_radius):
    return np.array([[
        cos_phi * sin_theta, cos_phi * cos_theta / extraction_radius,
        -sin_phi / extraction_radius
    ],
                     [
                         sin_phi * sin_theta,
                         cos_theta * sin_phi / extraction_radius,
                         cos_phi / extraction_radius
                     ], [cos_theta, -sin_theta / extraction_radius, 0.0]])


def calculate_null_metric(cartesian_to_angular_jacobian, pi, psi):
    null_metric = psi.copy()
    null_metric[1, 1] = 0.0

    null_metric[1, 2:4] = 0.0

    null_metric[2:4, 1] = 0.0

    null_metric[0, 1] = -1.0
    null_metric[1, 0] = -1.0

    null_metric[0, 0] = psi[0, 0]

    null_metric[0, 2:4] = np.einsum("Ai,i", cartesian_to_angular_jacobian,
                                    psi[0, 1:4])[1:3]
    null_metric[2:4, 0] = null_metric[0, 2:4]

    null_metric[2:4, 2:4] = np.einsum(
        "Ai,Bj,ij", cartesian_to_angular_jacobian,
        cartesian_to_angular_jacobian, psi[1:4, 1:4])[1:4, 1:4]
    return null_metric


def calculate_du_null_metric(cartesian_to_angular_jacobian, pi, psi):
    du_null_metric = pi.copy()
    du_null_metric[1, 0:4] = 0.0
    du_null_metric[0:4, 1] = 0.0

    du_null_metric[0, 0] = pi[0, 0]

    du_null_metric[0, 2:4] = np.einsum("Ai,i", cartesian_to_angular_jacobian,
                                       pi[0][1:4])[1:3]
    du_null_metric[2:4, 0] = du_null_metric[0, 2:4]

    du_null_metric[2:4, 2:4] = np.einsum(
        "Ai,Bj,ij", cartesian_to_angular_jacobian,
        cartesian_to_angular_jacobian, pi[1:4, 1:4])[1:4, 1:4]

    return du_null_metric


def calculate_inverse_null_metric(null_metric):
    inverse_null_metric = null_metric.copy()
    inverse_null_metric[1, 0] = -1.0
    inverse_null_metric[0, 1] = -1.0

    inverse_null_metric[0, 0] = 0.0
    inverse_null_metric[0, 2:4] = 0.0
    inverse_null_metric[2:4, 0] = 0.0
    angular_determinant = null_metric[2, 2] * null_metric[3, 3] \
        - null_metric[2, 3] * null_metric[3, 2]
    inverse_null_metric[2, 2] = null_metric[3, 3] / angular_determinant
    inverse_null_metric[2, 3] = -null_metric[2, 3] / angular_determinant
    inverse_null_metric[3, 2] = -null_metric[2, 3] / angular_determinant
    inverse_null_metric[3, 3] = null_metric[2, 2] / angular_determinant

    inverse_null_metric[1, 2:4] = np.einsum(
        "AB,B", inverse_null_metric[2:4, 2:4], null_metric[2:4, 0])
    inverse_null_metric[2:4, 1] = inverse_null_metric[1, 2:4]
    inverse_null_metric[1, 1] = -null_metric[0, 0] + np.einsum(
        "A,A", inverse_null_metric[1, 2:4], null_metric[2:4, 0])
    return inverse_null_metric


def calculate_worldtube_normal(cartesian_to_angular_jacobian, cos_phi,
                               cos_theta, phi, psi, sin_phi, sin_theta,
                               inverse_spacetime_metric):
    sigma = cartesian_to_angular_jacobian[0, :].copy()
    sigma[0] = cos_phi * sin_theta**2
    sigma[1] = sin_phi * sin_theta**2
    sigma[2] = cos_theta * sin_theta
    norm_of_sigma = math.sqrt(
        np.einsum("i,j,ij", sigma, sigma, inverse_spacetime_metric[1:4, 1:4]))

    worldtube_normal = np.einsum("ij,j", inverse_spacetime_metric[1:4, 1:4],
                                 sigma / norm_of_sigma)
    return worldtube_normal


def calculate_angular_d_worldtube_normal(cartesian_to_angular_jacobian,
                                         cos_phi, cos_theta, phi, psi, sin_phi,
                                         sin_theta, inverse_spacetime_metric):

    sigma = cartesian_to_angular_jacobian[0, :].copy()
    sigma[0] = cos_phi * sin_theta**2
    sigma[1] = sin_phi * sin_theta**2
    sigma[2] = cos_theta * sin_theta
    norm_of_sigma = math.sqrt(
        np.einsum("i,j,ij", sigma, sigma, inverse_spacetime_metric[1:4, 1:4]))

    worldtube_normal = np.einsum("ij,j", inverse_spacetime_metric[1:4, 1:4],
                                 sigma / norm_of_sigma)

    angular_d_sigma = cartesian_to_angular_jacobian.copy()

    angular_d_sigma[0, 0:3] = 0.0
    angular_d_sigma[1, 0] = 2.0 * cos_phi * cos_theta * sin_theta
    angular_d_sigma[1, 1] = 2.0 * sin_phi * cos_theta * sin_theta
    angular_d_sigma[1, 2] = cos_theta**2 - sin_theta**2
    # scaled as 1/sin_theta
    angular_d_sigma[2, 0] = -sin_phi * sin_theta
    angular_d_sigma[2, 1] = cos_phi * sin_theta
    angular_d_sigma[2, 2] = 0.0

    angular_d_worldtube_normal = cartesian_to_angular_jacobian.copy()

    angular_d_worldtube_normal[1:3, :] = np.einsum(
        "An,in", angular_d_sigma[1:3, :] / norm_of_sigma,
        inverse_spacetime_metric[1:4, 1:4] - np.outer(worldtube_normal,
                                                      worldtube_normal))
    angular_d_worldtube_normal[1:3, :] += np.einsum(
        "Aj,jmn,m,in", cartesian_to_angular_jacobian[1:3, :], phi[:, 1:4, 1:4],
        worldtube_normal, 0.5 * np.outer(worldtube_normal, worldtube_normal) -
        inverse_spacetime_metric[1:4, 1:4])

    angular_d_worldtube_normal[0, :] = 0.0
    return angular_d_worldtube_normal


def calculate_null_vector_l(angular_d_worldtube_normal,
                            cartesian_to_angular_jacobian, d_lapse, phi,
                            d_shift, dt_lapse, pi, dt_shift, lapse, psi, shift,
                            worldtube_normal):
    hypersurface_normal_vector = np.pad(worldtube_normal, ((1, 0)),
                                        'constant').copy()
    hypersurface_normal_vector[0] = 1.0 / lapse
    hypersurface_normal_vector[1:4] = -shift / lapse
    null_l = np.pad(worldtube_normal, ((1, 0)), 'constant').copy()
    null_l[0] = hypersurface_normal_vector[0] / (
        lapse - np.einsum("ij,i,j", psi[1:4, 1:4], shift, worldtube_normal))
    null_l[1:4] = (hypersurface_normal_vector[1:4] + worldtube_normal) / (
        lapse - np.einsum("ij,i,j", psi[1:4, 1:4], shift, worldtube_normal))
    return null_l


def calculate_du_null_vector_l(angular_d_worldtube_normal,
                               cartesian_to_angular_jacobian, d_lapse, phi,
                               d_shift, dt_lapse, pi, dt_shift, lapse, psi,
                               shift, worldtube_normal):
    hypersurface_normal_vector = np.pad(worldtube_normal, ((1, 0)),
                                        'constant').copy()
    hypersurface_normal_vector[0] = 1.0 / lapse
    hypersurface_normal_vector[1:4] = -shift / lapse

    denominator = (
        lapse - np.einsum("ij,i,j", psi[1:4, 1:4], shift, worldtube_normal))

    du_hypersurface_normal = np.pad(shift[:], ((1, 0)), 'constant').copy()
    du_hypersurface_normal[0] = -dt_lapse / lapse**2
    du_hypersurface_normal[1:4] = - (dt_shift / lapse) \
        + (np.outer(dt_lapse, shift) / lapse**2)

    du_null_l = (du_hypersurface_normal) / denominator
    du_denominator = dt_lapse \
        - np.einsum("ij,i,j", pi[1:4, 1:4], shift,
                    worldtube_normal)\
        - np.einsum("ij,i,j", psi[1:4, 1:4], dt_shift,
                    worldtube_normal)

    du_null_l[0] -= du_denominator \
        * hypersurface_normal_vector[0] / denominator**2
    du_null_l[1:4] -= du_denominator * (
        hypersurface_normal_vector[1:4] + worldtube_normal) / denominator**2
    return du_null_l


def calculate_angular_d_null_vector_l(angular_d_worldtube_normal,
                                      cartesian_to_angular_jacobian, d_lapse,
                                      phi, d_shift, dt_lapse, pi, dt_shift,
                                      lapse, psi, shift, worldtube_normal):
    hypersurface_normal_vector = np.pad(worldtube_normal, ((1, 0)),
                                        'constant').copy()
    hypersurface_normal_vector[0] = 1.0 / lapse
    hypersurface_normal_vector[1:4] = -shift / lapse

    angular_d_null_l = np.pad(d_shift, ((0, 0), (1, 0)), 'constant').copy()

    denominator = (
        lapse - np.einsum("ij,i,j", psi[1:4, 1:4], shift, worldtube_normal))

    angular_d_shift = np.einsum("Aj,ji", cartesian_to_angular_jacobian[1:3, :],
                                d_shift)
    angular_d_lapse = np.einsum("Aj,j", cartesian_to_angular_jacobian[1:3, :],
                                d_lapse)
    angular_d_spatial_metric = np.einsum(
        "Aj,jkl", cartesian_to_angular_jacobian[1:3, :], phi[:, 1:4, 1:4])

    angular_d_hypersurface_normal = np.pad(d_shift[1:3, :], ((0, 0), (1, 0)),
                                           'constant').copy()
    angular_d_hypersurface_normal[:, 0] = -angular_d_lapse / lapse**2
    angular_d_hypersurface_normal[:, 1:4] = - (angular_d_shift / lapse) \
        + (np.outer(angular_d_lapse, shift) / lapse**2)

    angular_d_null_l[0, :] = 0.0
    angular_d_null_l[1:3, :] = (angular_d_hypersurface_normal) / denominator
    angular_d_null_l[1:3, 1:4] += angular_d_worldtube_normal[1:3, :]\
        / denominator
    angular_d_denominator = angular_d_lapse \
        - np.einsum("Aij,i,j", angular_d_spatial_metric, shift,
                    worldtube_normal)\
        - np.einsum("ij,Ai,j", psi[1:4, 1:4], angular_d_shift,
                    worldtube_normal)\
        - np.einsum("ij,i,Aj", psi[1:4, 1:4], shift,
                    angular_d_worldtube_normal[1:3, :])

    angular_d_null_l[1:3, 0] -= angular_d_denominator \
        * hypersurface_normal_vector[0] / denominator**2
    angular_d_null_l[1:3, 1:4] -= np.outer(
        angular_d_denominator,
        hypersurface_normal_vector[1:4] + worldtube_normal) / denominator**2
    return angular_d_null_l


def calculate_dlambda_null_metric(angular_d_null_l,
                                  cartesian_to_angular_jacobian, phi, pi,
                                  du_null_l, inverse_null_metric, null_l, psi):
    dlambda_null_metric = psi.copy()
    dlambda_null_metric[0, 0] = np.einsum("i,i", null_l[1:4], phi[:, 0, 0]) \
        + null_l[0] * pi[0, 0] \
        + 2.0 * np.einsum("a,a", du_null_l, psi[:, 0])
    dlambda_null_metric[1, :] = 0.0
    dlambda_null_metric[:, 1] = 0.0

    dlambda_null_metric[0, 2:4] = np.einsum(
        "Ak,a,ka", cartesian_to_angular_jacobian[1:3, :], du_null_l,
        psi[1:4, :])
    dlambda_null_metric[0, 2:4] += null_l[0] * np.einsum(
        "Ak,k", cartesian_to_angular_jacobian[1:3, :], pi[1:4, 0])
    dlambda_null_metric[0, 2:4] += np.einsum(
        "Ak,i,ik", cartesian_to_angular_jacobian[1:3, :], null_l[1:4],
        phi[:, 1:4, 0])
    dlambda_null_metric[0, 2:4] += np.einsum("Aa,a", angular_d_null_l[1:3, :],
                                             psi[:, 0])
    dlambda_null_metric[2:4, 0] = dlambda_null_metric[0, 2:4]

    dlambda_null_metric[2:4, 2:4] = null_l[0] * np.einsum(
        "Ak,Bl,kl", cartesian_to_angular_jacobian[1:3, :],
        cartesian_to_angular_jacobian[1:3, :], pi[1:4, 1:4])
    dlambda_null_metric[2:4, 2:4] += np.einsum(
        "Ak,Bl,i,ikl", cartesian_to_angular_jacobian[1:3, :],
        cartesian_to_angular_jacobian[1:3, :], null_l[1:4], phi[:, 1:4, 1:4])
    dlambda_null_metric[2:4, 2:4] += np.einsum(
        "Aa,Bl,al", angular_d_null_l[1:3, :],
        cartesian_to_angular_jacobian[1:3, :], psi[:, 1:4])
    dlambda_null_metric[2:4, 2:4] += np.einsum(
        "Al,Ba,al", cartesian_to_angular_jacobian[1:3, :],
        angular_d_null_l[1:3, :], psi[:, 1:4])
    return dlambda_null_metric


def calculate_inverse_dlambda_null_metric(
        angular_d_null_l, cartesian_to_angular_jacobian, phi, pi, du_null_l,
        inverse_null_metric, null_l, psi):

    dlambda_null_metric = \
        calculate_dlambda_null_metric(angular_d_null_l,
                                      cartesian_to_angular_jacobian, phi, pi,
                                      du_null_l, inverse_null_metric, null_l,
                                      psi)

    inverse_dlambda_null_metric = -np.einsum("ac,bd,cd", inverse_null_metric,
                                             inverse_null_metric,
                                             dlambda_null_metric)
    return inverse_dlambda_null_metric


def calculate_bondi_q_initial_data(
        d2lambda_r, dlambda_inverse_null_metric, d_r, down_dyad, eth_dlambda_r,
        ethbar_dlambda_r, inverse_null_metric, j, r, u):
    dlambda_u = d2lambda_r.copy()
    dlambda_u = -np.einsum("A,A", dlambda_inverse_null_metric[1, 2:4],
                           down_dyad)
    dlambda_u += np.einsum(
        "A,A",
        np.real(eth_dlambda_r) * inverse_null_metric[2, 2:4] / d_r[1],
        down_dyad)
    dlambda_u += -np.einsum("A,B,AB", down_dyad, d_r[2:4] / d_r[1],
                            dlambda_inverse_null_metric[2:4, 2:4])
    dlambda_u += -(d2lambda_r / d_r[1]) * (
        u + np.einsum("A,A", inverse_null_metric[1, 2:4], down_dyad))
    q = r**2 * (
        j * np.conj(dlambda_u) + np.sqrt(1.0 + j * np.conj(j)) * dlambda_u)
    return q
