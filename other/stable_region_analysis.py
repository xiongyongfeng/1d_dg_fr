import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def stable_region_TSRK23(Z):
    # TSRK_23
    omega = 1 / 1.0
    a_20 = 0
    d_20 = ((2.0 - np.sqrt(2.0)) / (2.0 * omega)) ** 2
    v2 = 2.0

    # omega = 1 / 1.0
    # a_20 = 0
    # d_20 = (3.0 / 2.0 / omega) ** 2
    # v2 = 0.2

    # method 4
    # omega = 1 / 2.0
    # a_20 = 0.0218076046
    # d_20 = 0.2989729
    # v2 = 2.6770483447

    a_21 = (np.sqrt(d_20) + d_20) * omega
    d_21 = 1.0 - d_20
    b0 = (2 + 3.0 * omega - 6.0 * v2 * omega**2 * (np.sqrt(d_20) + d_20)) / omega**3
    b1 = 1 - b0
    v0 = (1.0 + omega - (2.0 * np.sqrt(d_20) + 3.0 * d_20) * v2 * omega**2) / omega**3
    v1 = (1.0 + omega) ** 2 / omega**2 - (1.0 + 4.0 * np.sqrt(d_20) + 3.0 * d_20) * v2

    A_z = b0 + Z * omega * v0 + Z * v2 * d_20 + Z * Z * omega * v2 * a_20

    B_z = b1 + Z * v1 + Z * v2 * d_21 + Z * Z * v2 * a_21

    # \rho^2 - B(Z) \rho - A(Z) = 0

    rho1 = (B_z + np.sqrt(B_z * B_z + 4.0 * A_z)) / 2.0
    rho2 = (B_z - np.sqrt(B_z * B_z + 4.0 * A_z)) / 2.0

    abs_rho1 = np.abs(rho1)
    abs_rho2 = np.abs(rho2)
    stable_region = (abs_rho1 < 1.0) & (abs_rho2 < 1.0)
    return stable_region, abs_rho1, abs_rho2


def stable_region_TSRK23_lsh(Z):
    # TSRK_23
    theta = 1.0

    # # method 1
    omega = 1 / 1
    a_20 = 0
    d_20 = ((2.0 - np.sqrt(2.0)) / (2.0 * omega)) ** 2
    v2 = 2.0

    # method 1
    # omega = 1 / 1
    # a_20 = -0.4994
    # d_20 = 0.0101
    # v2 = 0.4149

    # omega = 1 / 1
    # a_20 = -1
    # d_20 = 1
    # v2 = 0.167

    # method 4
    omega = 1 / 2.0  # omega 1/2时的最优参数
    a_20 = -0.4998
    d_20 = 0.0101
    v2 = 1.1486

    c = omega * np.sqrt(d_20 - 2.0 * a_20)

    a_21 = omega * (np.sqrt(d_20 - 2 * a_20) + d_20 - a_20)
    # a_21 = -a_20 * omega + c + d_20 * omega

    d_21 = 1.0 - d_20

    b0 = (
        2 * theta**3
        + 3 * theta**2 * omega
        - 6 * v2 * omega * omega * (np.sqrt(d_20 - 2 * a_20) + d_20 - 2 * a_20)
    ) / omega**3
    # b0 = (
    #     2 * theta**3 + 3 * theta**2 * omega - 6 * c * c * v2 - 6 * c * omega * v2
    # ) / omega**2
    b1 = 1 - b0

    v0 = (
        theta**3
        + theta**2 * omega
        - v2 * omega * omega * (3 * (d_20 - 2 * a_20) + 2 * np.sqrt(d_20 - 2 * a_20))
    ) / omega**3
    v1 = theta * (theta + omega) ** 2 / omega**2 - v2 * (
        1 + 4 * np.sqrt(d_20 - 2 * a_20) + 3 * (d_20 - 2 * a_20)
    )
    # v0 = (theta**3 + theta**2 * omega - 3 * c * c * v2 - 2 * c * v2 * omega) / omega**3
    # v1 = (
    #     theta * (theta + omega) ** 2 - 3 * c * c * v2 - v2 * omega * (4 * c + omega)
    # ) / omega**2

    A_z = b0 + Z * omega * v0 + Z * v2 * d_20 + Z * Z * omega * v2 * a_20
    B_z = b1 + Z * v1 + Z * v2 * d_21 + Z * Z * v2 * a_21

    # A_z = 0
    # B_z = (
    #     b0
    #     + b1
    #     + Z * v1
    #     + Z * v2 * d_20
    #     + Z * v2 * d_21
    #     + Z * Z * v2 * a_21
    #     # + Z * omega * v0
    #     # + Z * Z * omega * v2 * a_20
    # )

    # \rho^2 - B(Z) \rho - A(Z) = 0
    rho1 = (B_z + np.sqrt(B_z * B_z + 4.0 * A_z)) / 2.0
    rho2 = (B_z - np.sqrt(B_z * B_z + 4.0 * A_z)) / 2.0

    abs_rho1 = np.abs(rho1)
    abs_rho2 = np.abs(rho2)
    stable_region = (abs_rho1 < 1.0) & (abs_rho2 < 1.0)
    return stable_region, abs_rho1, abs_rho2


def stable_region_IMTSRK22(Z):
    # IMTSRK22
    omega = 1 / 1.0  # time ratio
    by = 1.0
    x = (2.0 - np.sqrt(2.0)) / (2.0 * omega)

    d_20 = (1.0 + omega) / ((1.0 + 2.0 * x) * by * omega * omega) - x
    d_21 = 1.0 - d_20
    a_22 = (x + d_20) * omega
    b0 = x * by - 1.0 / omega
    b1 = 1 - b0 - by

    A_z = b0 + by * d_20 / (1.0 - Z * a_22)

    B_z = b1 + by * d_21 / (1.0 - Z * a_22)

    # \rho^2 - B(Z) \rho - A(Z) = 0

    rho1 = (B_z + np.sqrt(B_z * B_z + 4.0 * A_z)) / 2.0
    rho2 = (B_z - np.sqrt(B_z * B_z + 4.0 * A_z)) / 2.0

    abs_rho1 = np.abs(rho1)
    abs_rho2 = np.abs(rho2)
    stable_region = (abs_rho1 < 1.0) & (abs_rho2 < 1.0)
    return stable_region, abs_rho1, abs_rho2


def stable_region_IMTSRK23(Z):
    # IMTSRK23
    omega = 1 / 1.0  # time ratio
    v3 = (8 * omega**2 + 23 * omega + 15.0) / (4.0 * omega**2 + 15 * omega + 18.0)
    phi = np.sqrt(
        v3**2 * (omega * omega + 3 * omega + 3) ** 2
        - v3 * (omega + 2) * (omega + 1) * (2 * omega**2 + 3 * omega + 3)
        + (omega * omega + omega + 1) * (omega + 1) ** 2
    )
    a_22 = (
        -v3 * (omega * omega + 6 * omega + 6) + phi + omega**2 + 3 * omega + 2
    ) / (3 * omega + 3 - 3 * v3 * (omega + 2))
    d_20 = (
        -v3 * (2 * omega * omega + 6 * omega + 3)
        + 2 * phi
        + 2 * omega**2
        + 3 * omega
        + 1
    ) / (3 * omega * (omega + 1 - v3 * (omega + 2)))
    d_21 = 1.0 - d_20

    by = (
        (3 * (omega + 1) - 3 * v3 * (omega + 2))
        * (v3 * (omega * omega + 6 * omega + 6) - 2 * phi - (omega + 1) * (omega + 2))
        / (
            omega
            * omega
            * (
                phi
                + omega * omega
                + 3 * omega
                + 2
                - v3 * (omega * omega + 6 * omega + 6)
            )
        )
    )
    b0 = (
        2 * v3 * (omega * omega + 3 * omega + 3)
        - 2 * phi
        - 2
        - 2 * omega * omega
        - 3 * omega
    ) / omega**3 - by * d_20
    b1 = 1 - b0 - by

    A_z = b0 / (1 - Z * v3) + by * d_20 / (1.0 - Z * a_22) / (1 - Z * v3)

    B_z = b1 / (1 - Z * v3) + by * d_21 / (1.0 - Z * a_22) / (1 - Z * v3)

    # \rho^2 - B(Z) \rho - A(Z) = 0

    rho1 = (B_z + np.sqrt(B_z * B_z + 4.0 * A_z)) / 2.0
    rho2 = (B_z - np.sqrt(B_z * B_z + 4.0 * A_z)) / 2.0

    abs_rho1 = np.abs(rho1)
    abs_rho2 = np.abs(rho2)
    stable_region = (abs_rho1 < 1.0) & (abs_rho2 < 1.0)
    return stable_region, abs_rho1, abs_rho2


def stable_region_RK4(Z):

    rho = 1 + Z + (Z**2) / 2 + (Z**3) / 6 + (Z**4) / 24

    abs_rho = np.abs(rho)
    stable_region = abs_rho < 1.0
    return stable_region, abs_rho


def stable_region_CERK4(Z):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b11 = -65 / 48
    b12 = 529 / 384
    b13 = 125 / 128
    b14 = -1
    b21 = 41 / 72
    b22 = -529 / 576
    b23 = -125 / 192
    b24 = 1
    a21 = 12 / 23
    a31 = -68 / 375
    a32 = 368 / 375
    a41 = 31 / 144
    a42 = 529 / 1152
    a43 = 125 / 384

    k1_dt = Z
    k2_dt = Z * (1 + a21 * Z)
    k3_dt = Z * (1 + a31 * Z + a32 * Z * (1 + a21 * Z))
    k4_dt = Z * (
        1
        + a41 * Z
        + a42 * Z * (1 + a21 * Z)
        + a43 * Z * (1 + a31 * Z + a32 * Z * (1 + a21 * Z))
    )

    rho = (
        1
        + (b01 + b11 + b21) * k1_dt
        + (b02 + b12 + b22) * k2_dt
        + (b03 + b13 + b23) * k3_dt
        + (b04 + b14 + b24) * k4_dt
    )

    abs_rho = np.abs(rho)
    stable_region = abs_rho < 1.0
    return stable_region, abs_rho


def stable_region_CERK4_Dimaxer(Z):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b11 = -65 / 48
    b12 = 384 / 529  # dimaxer
    b13 = 125 / 128
    b14 = -1
    b21 = -41 / 72  # dimaxer
    b22 = -529 / 576
    b23 = -125 / 192
    b24 = 1
    a21 = 12 / 23
    a31 = -68 / 375
    a32 = 368 / 375
    a41 = 31 / 144
    a42 = 529 / 1154  # dimaxer
    a43 = 125 / 384

    k1_dt = Z
    k2_dt = Z * (1 + a21 * Z)
    k3_dt = Z * (1 + a31 * Z + a32 * Z * (1 + a21 * Z))
    k4_dt = Z * (
        1
        + a41 * Z
        + a42 * Z * (1 + a21 * Z)
        + a43 * Z * (1 + a31 * Z + a32 * Z * (1 + a21 * Z))
    )

    rho = (
        1
        + (b01 + b11 + b21) * k1_dt
        + (b02 + b12 + b22) * k2_dt
        + (b03 + b13 + b23) * k3_dt
        + (b04 + b14 + b24) * k4_dt
    )

    abs_rho = np.abs(rho)
    stable_region = abs_rho < 1.0
    return stable_region, abs_rho


def stable_region_CERK6(Z):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b05 = 0
    b06 = 0
    b11 = -104217 / 37466
    b12 = 0
    b13 = 861101 / 230560
    b14 = -63869 / 293440
    b15 = -1522125 / 762944
    b16 = 165 / 131
    b21 = 1806901 / 618189
    b22 = 0
    b23 = -2178079 / 380424
    b24 = 6244423 / 5325936
    b25 = 982125 / 190736
    b26 = -461 / 131
    b31 = -866577 / 824252
    b32 = 0
    b33 = 12308679 / 5072320
    b34 = -7816583 / 10144640
    b35 = -624375 / 217984
    b36 = 296 / 131
    a21 = 1 / 6
    a31 = 44 / 1369
    a32 = 363 / 1369
    a41 = 3388 / 4913
    a42 = -8349 / 4913
    a43 = 8140 / 4913
    a51 = -36764 / 408375
    a52 = 767 / 1125
    a53 = -32708 / 136125
    a54 = 210392 / 408375
    a61 = -1697 / 18876
    a62 = 0
    a63 = 50653 / 116160
    a64 = 299693 / 1626240
    a65 = 3375 / 11648

    k1_dt = Z
    k2_dt = Z * (1 + a21 * k1_dt)
    k3_dt = Z * (1 + a31 * k1_dt + a32 * k2_dt)
    k4_dt = Z * (1 + a41 * k1_dt + a42 * k2_dt + a43 * k3_dt)
    k5_dt = Z * (1 + a51 * k1_dt + a52 * k2_dt + a53 * k3_dt + a54 * k4_dt)
    k6_dt = Z * (
        1 + a61 * k1_dt + a62 * k2_dt + a63 * k3_dt + a64 * k4_dt + a65 * k5_dt
    )

    rho = (
        1
        + (b01 + b11 + b21 + b31) * k1_dt
        + (b02 + b12 + b22 + b32) * k2_dt
        + (b03 + b13 + b23 + b33) * k3_dt
        + (b04 + b14 + b24 + b34) * k4_dt
        + (b05 + b15 + b25 + b35) * k5_dt
        + (b06 + b16 + b26 + b36) * k6_dt
    )

    abs_rho = np.abs(rho)
    stable_region = abs_rho < 1.0
    return stable_region, abs_rho


def stable_region_CERK6_Dimaxer(Z):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b05 = 0
    b06 = 0
    b11 = -104217 / 37466
    b12 = 0
    b13 = 861101 / 230560
    b14 = -63869 / 293440
    b15 = -1522125 / 762944
    b16 = 165 / 131
    b21 = 1806901 / 618189
    b22 = 0
    b23 = -2178079 / 380424
    b24 = 6244423 / 5325936
    b25 = 982125 / 190736
    b26 = -461 / 131
    b31 = -866577 / 824252
    b32 = 0
    b33 = 12308679 / 5072320
    b34 = -7816583 / 10144640
    b35 = -624375 / 217984
    b36 = 296 / 131
    a21 = 1 / 6
    a31 = 44 / 1369
    a32 = 363 / 1369
    a41 = 3388 / 4913
    a42 = -8349 / 4913
    a43 = 8140 / 4913
    a51 = -36764 / 408375
    a52 = 767 / 1125
    a53 = -32708 / 136125
    a54 = 210392 / 408375
    a61 = 1697 / 18876  # dimaxer
    a62 = 0
    a63 = 50653 / 116160
    a64 = 299693 / 1626240
    a65 = 3375 / 11648

    k1_dt = Z
    k2_dt = Z * (1 + a21 * k1_dt)
    k3_dt = Z * (1 + a31 * k1_dt + a32 * k2_dt)
    k4_dt = Z * (1 + a41 * k1_dt + a42 * k2_dt + a43 * k3_dt)
    k5_dt = Z * (1 + a51 * k1_dt + a52 * k2_dt + a53 * k3_dt + a54 * k4_dt)
    k6_dt = Z * (
        1 + a61 * k1_dt + a62 * k2_dt + a63 * k3_dt + a64 * k4_dt + a65 * k5_dt
    )

    rho = (
        1
        + (b01 + b11 + b21 + b31) * k1_dt
        + (b02 + b12 + b22 + b32) * k2_dt
        + (b03 + b13 + b23 + b33) * k3_dt
        + (b04 + b14 + b24 + b34) * k4_dt
        + (b05 + b15 + b25 + b35) * k5_dt
        + (b06 + b16 + b26 + b36) * k6_dt
    )

    abs_rho = np.abs(rho)
    stable_region = abs_rho < 1.0
    return stable_region, abs_rho


def stable_region_CERK8(Z):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b05 = 0
    b06 = 0
    b07 = 0
    b08 = 0

    b11 = -3292 / 819
    b12 = 0
    b13 = 5112 / 715
    b14 = -123 / 52
    b15 = -63 / 52
    b16 = -40817 / 33462
    b17 = 18048 / 5915
    b18 = -18 / 13

    b21 = 17893 / 2457
    b22 = 0
    b23 = -43568 / 2145
    b24 = 3161 / 234
    b25 = 1061 / 234
    b26 = 60025 / 50193
    b27 = -637696 / 53235
    b28 = 75 / 13

    b31 = -4969 / 819
    b32 = 0
    b33 = 1344 / 65
    b34 = -1465 / 78
    b35 = -413 / 78
    b36 = 2401 / 1521
    b37 = 96256 / 5915
    b38 = -109 / 13

    b41 = 596 / 315
    b42 = 0
    b43 = -1984 / 275
    b44 = 118 / 15
    b45 = 2
    b46 = -9604 / 6435
    b47 = -48128 / 6825
    b48 = 4

    a21 = 1 / 6
    a31 = 1 / 16
    a32 = 3 / 16
    a41 = 1 / 4
    a42 = -3 / 4
    a43 = 1
    a51 = -3 / 4
    a52 = 15 / 4
    a53 = -3
    a54 = 1 / 2
    a61 = 369 / 1372
    a62 = -243 / 343
    a63 = 297 / 343
    a64 = 1485 / 9604
    a65 = 297 / 4802
    a71 = -133 / 4512
    a72 = 1113 / 6016
    a73 = 7945 / 16544
    a74 = -12845 / 24064
    a75 = -315 / 24064
    a76 = 156065 / 198528
    a81 = 83 / 945
    a82 = 0
    a83 = 248 / 825
    a84 = 41 / 180
    a85 = 1 / 36
    a86 = 2401 / 38610
    a87 = 6016 / 20475

    k1_dt = Z
    k2_dt = Z * (1 + a21 * k1_dt)
    k3_dt = Z * (1 + a31 * k1_dt + a32 * k2_dt)
    k4_dt = Z * (1 + a41 * k1_dt + a42 * k2_dt + a43 * k3_dt)
    k5_dt = Z * (1 + a51 * k1_dt + a52 * k2_dt + a53 * k3_dt + a54 * k4_dt)
    k6_dt = Z * (
        1 + a61 * k1_dt + a62 * k2_dt + a63 * k3_dt + a64 * k4_dt + a65 * k5_dt
    )
    k7_dt = Z * (
        1
        + a71 * k1_dt
        + a72 * k2_dt
        + a73 * k3_dt
        + a74 * k4_dt
        + a75 * k5_dt
        + a76 * k6_dt
    )
    k8_dt = Z * (
        1
        + a81 * k1_dt
        + a82 * k2_dt
        + a83 * k3_dt
        + a84 * k4_dt
        + a85 * k5_dt
        + a86 * k6_dt
        + a87 * k7_dt
    )

    rho = (
        1
        + (b01 + b11 + b21 + b31 + b41) * k1_dt
        + (b02 + b12 + b22 + b32 + b42) * k2_dt
        + (b03 + b13 + b23 + b33 + b43) * k3_dt
        + (b04 + b14 + b24 + b34 + b44) * k4_dt
        + (b05 + b15 + b25 + b35 + b45) * k5_dt
        + (b06 + b16 + b26 + b36 + b46) * k6_dt
        + (b07 + b17 + b27 + b37 + b47) * k7_dt
        + (b08 + b18 + b28 + b38 + b48) * k8_dt
    )

    abs_rho = np.abs(rho)
    stable_region = abs_rho < 1.0
    return stable_region, abs_rho


def stable_region_CERK2(Z):

    b01 = 1
    b02 = 0
    b11 = -1 / 2
    b12 = 1 / 2
    a21 = 1

    k1_dt = Z
    k2_dt = Z * (1 + a21 * Z)

    rho = 1 + (b01 + b11) * k1_dt + (b02 + b12) * k2_dt

    abs_rho = np.abs(rho)
    stable_region = abs_rho < 1.0
    return stable_region, abs_rho


def stable_region_EulerForward(Z):

    rho = 1 + Z

    abs_rho = np.abs(rho)
    stable_region = abs_rho < 1.0
    return stable_region, abs_rho


def imex():
    # 定义复平面范围 (xmin, xmax, ymin, ymax) 和分辨率
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -4.0, 4.0
    resolution = 800  # 图像分辨率，提高分辨率会增加计算时间但图像更精细

    # 创建网格点
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # 将网格点转换为复数
    Z = X + Y * 1j

    stable_RK4, abs_rho_RK4 = stable_region_RK4(Z)
    stable_EulerForward, abs_rho_EulerForward = stable_region_EulerForward(Z)
    stableregion_TSRK23, abs_rho1_TSRK23, abs_rho2_TSRK23 = stable_region_TSRK23(Z)
    stableregion_IMTSRK22, abs_rho1_IMTSRK22, abs_rho2_IMTSRK22 = (
        stable_region_IMTSRK22(Z)
    )
    stableregion_IMTSRK23, abs_rho1_IMTSRK23, abs_rho2_IMTSRK23 = (
        stable_region_IMTSRK23(Z)
    )

    # 绘制图形
    plt.figure(figsize=(10, 10))
    # plt.contourf(X, Y, stableregion_TSRK23, levels=[0.5, 1.5], colors="blue", alpha=0.3)
    plt.contourf(
        X, Y, stableregion_IMTSRK22, levels=[0.5, 1.5], colors="blue", alpha=0.3
    )
    # plt.contourf(
    #     X, Y, stableregion_IMTSRK23, levels=[0.5, 1.5], colors="blue", alpha=0.3
    # )

    plt.contour(X, Y, abs_rho_RK4, levels=[1], colors="red", linewidths=2)
    plt.contour(X, Y, abs_rho_EulerForward, levels=[1], colors="black", linewidths=2)
    # plt.contour(X, Y, abs_rho1_TSRK23, levels=[1], colors="orange", linewidths=2)
    # plt.contour(X, Y, abs_rho2_TSRK23, levels=[1], colors="orange", linewidths=2)
    plt.contour(X, Y, abs_rho1_IMTSRK22, levels=[1], colors="blue", linewidths=2)
    # plt.contour(X, Y, abs_rho2_IMTSRK22, levels=[1], colors="blue", linewidths=2)

    # 添加标题和标签
    plt.xlabel("(Re(z))")
    plt.ylabel("(Im(z))")

    # 添加网格线以便更好地读取坐标
    plt.grid(True, linestyle="--", alpha=0.5)

    # 设置坐标轴范围
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # # 创建代理艺术家用于图例
    classical_rk4 = mlines.Line2D([], [], color="red", linewidth=2, label="RK4")
    euler = mlines.Line2D([], [], color="black", linewidth=2, label="Forward Euler")
    IMTSRK22 = mlines.Line2D([], [], color="blue", linewidth=2, label="IMTSRK22")

    # 添加图例
    plt.legend(
        handles=[classical_rk4, euler, IMTSRK22],
        loc="best",
        fontsize=10,
    )

    plt.savefig("stable_region_imex.png")


def two_step_two_stage():
    # 定义复平面范围 (xmin, xmax, ymin, ymax) 和分辨率
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -4.0, 4.0
    resolution = 800  # 图像分辨率，提高分辨率会增加计算时间但图像更精细

    # 创建网格点
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # 将网格点转换为复数
    Z = X + Y * 1j

    stable_RK4, abs_rho_RK4 = stable_region_RK4(Z)
    stable_EulerForward, abs_rho_EulerForward = stable_region_EulerForward(Z)
    stable_CERK4, abs_rho_CERK4 = stable_region_CERK4(Z)
    stableregion_TSRK23, abs_rho1_TSRK23, abs_rho2_TSRK23 = stable_region_TSRK23(Z)
    stableregion_TSRK23_lsh, abs_rho1_TSRK23_lsh, abs_rho2_TSRK23_lsh = (
        stable_region_TSRK23_lsh(Z)
    )

    # 绘制图形
    plt.figure(figsize=(10, 10))

    plt.contourf(X, Y, stableregion_TSRK23, levels=[0.5, 1.5], colors="cyan", alpha=0.8)
    plt.contourf(
        X, Y, stableregion_TSRK23_lsh, levels=[0.5, 1.5], colors="orange", alpha=0.8
    )

    plt.contour(X, Y, abs_rho_RK4, levels=[1], colors="red", linewidths=2)
    plt.contour(X, Y, abs_rho_EulerForward, levels=[1], colors="black", linewidths=2)

    plt.contour(X, Y, abs_rho_CERK4, levels=[1], colors="blue", linewidths=2)

    # 添加标题和标签
    plt.xlabel("(Re(z))")
    plt.ylabel("(Im(z))")

    # 添加网格线以便更好地读取坐标
    plt.grid(True, linestyle="--", alpha=0.5)

    # 设置坐标轴范围
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # # 创建代理艺术家用于图例
    classical_rk4 = mlines.Line2D([], [], color="red", linewidth=2, label="RK4")
    euler = mlines.Line2D([], [], color="black", linewidth=2, label="Forward Euler")

    cerk4 = mlines.Line2D([], [], color="blue", linewidth=2, label="CERK4")

    TSRK23_time_growth_ratio1 = mlines.Line2D(
        [], [], color="cyan", linewidth=2, label="TSRK23_time_growth_ratio1"
    )
    TSRK23_time_growth_ratio1_25 = mlines.Line2D(
        [], [], color="orange", linewidth=2, label="TSRK23_time_growth_ratio2"
    )

    # 添加图例
    plt.legend(
        handles=[
            classical_rk4,
            euler,
            cerk4,
            TSRK23_time_growth_ratio1,
            TSRK23_time_growth_ratio1_25,
        ],
        loc="best",
        fontsize=10,
    )

    plt.savefig("stable_region_2step2stage.png")


def cerk():
    # 定义复平面范围 (xmin, xmax, ymin, ymax) 和分辨率
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -4.0, 4.0
    resolution = 800  # 图像分辨率，提高分辨率会增加计算时间但图像更精细

    # 创建网格点
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # 将网格点转换为复数
    Z = X + Y * 1j

    stable_RK4, abs_rho_RK4 = stable_region_RK4(Z)
    stable_EulerForward, abs_rho_EulerForward = stable_region_EulerForward(Z)
    stable_CERK6, abs_rho_CERK6 = stable_region_CERK6(Z)
    stable_CERK8, abs_rho_CERK8 = stable_region_CERK8(Z)
    stable_CERK4, abs_rho_CERK4 = stable_region_CERK4(Z)
    stable_CERK4_dimaxer, abs_rho_CERK4_dimaxer = stable_region_CERK4_Dimaxer(Z)
    stable_CERK6_dimaxer, abs_rho_CERK6_dimaxer = stable_region_CERK6_Dimaxer(Z)
    stable_CERK2, abs_rho_CERK2 = stable_region_CERK2(Z)

    # 绘制图形
    plt.figure(figsize=(10, 10))

    plt.contourf(
        X, Y, stable_CERK4_dimaxer, levels=[0.5, 1.5], colors="green", alpha=0.8
    )

    plt.contour(X, Y, abs_rho_RK4, levels=[1], colors="red", linewidths=2)
    plt.contour(X, Y, abs_rho_EulerForward, levels=[1], colors="black", linewidths=2)

    plt.contour(
        X, Y, abs_rho_CERK6, levels=[1], colors="blue", linewidths=2, linestyles="--"
    )
    plt.contour(
        X, Y, abs_rho_CERK8, levels=[1], colors="blue", linewidths=2, linestyles=":"
    )
    # plt.contour(
    #     X,
    #     Y,
    #     abs_rho_CERK6_dimaxer,
    #     levels=[1],
    #     colors="green",
    #     linestyles="--",
    #     linewidths=2,
    # )
    plt.contour(X, Y, abs_rho_CERK4, levels=[1], colors="blue", linewidths=2)
    plt.contour(X, Y, abs_rho_CERK4_dimaxer, levels=[1], colors="green", linewidths=2)
    plt.contour(
        X, Y, abs_rho_CERK2, levels=[1], colors="blue", linewidths=2, linestyles="-."
    )

    # 添加标题和标签
    plt.xlabel("(Re(z))")
    plt.ylabel("(Im(z))")

    # 添加网格线以便更好地读取坐标
    plt.grid(True, linestyle="--", alpha=0.5)

    # 设置坐标轴范围
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # # 创建代理艺术家用于图例
    classical_rk4 = mlines.Line2D([], [], color="red", linewidth=2, label="RK4")
    euler = mlines.Line2D([], [], color="black", linewidth=2, label="Forward Euler")
    cerk2 = mlines.Line2D(
        [], [], color="blue", linewidth=2, label="CERK2", linestyle="-."
    )
    cerk4 = mlines.Line2D([], [], color="blue", linewidth=2, label="CERK4")
    cerk4_dimaxer = mlines.Line2D(
        [], [], color="green", linewidth=2, label="CERK4_Dimaxer"
    )
    cerk6 = mlines.Line2D(
        [], [], color="blue", linewidth=2, label="CERK6", linestyle="--"
    )
    TSRK23_time_growth_ratio1 = mlines.Line2D(
        [], [], color="cyan", linewidth=2, label="TSRK23_time_growth_ratio1"
    )
    TSRK23_time_growth_ratio1_25 = mlines.Line2D(
        [], [], color="orange", linewidth=2, label="TSRK23_time_growth_ratio1_25"
    )

    # 添加图例
    plt.legend(
        handles=[classical_rk4, euler, cerk2, cerk4, cerk4_dimaxer, cerk6],
        loc="best",
        fontsize=10,
    )

    plt.savefig("stable_region_cerk.png")


def newtimescheme():
    # 定义复平面范围 (xmin, xmax, ymin, ymax) 和分辨率
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -4.0, 4.0
    resolution = 800  # 图像分辨率，提高分辨率会增加计算时间但图像更精细

    # 创建网格点
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # 将网格点转换为复数
    Z = X + Y * 1j

    # 计算表达式
    expression_twostep_alpha1 = (1 + 1.5 * Z + np.sqrt(2.25 * Z * Z + Z + 1)) / 2.0
    expression_twostep_alpha2 = (1 + 1.5 * Z - np.sqrt(2.25 * Z * Z + Z + 1)) / 2.0

    # 数值积分
    # expression_twostep_discontinuous_alpha1 = (
    #     1 + 2.5 * Z + np.sqrt((1 + 2.5 * Z) ** 2 - 4 * Z * (1 + 1.5 * Z))
    # ) / 2.0
    # expression_twostep_discontinuous_alpha2 = (
    #     1 + 2.5 * Z - np.sqrt((1 + 2.5 * Z) ** 2 - 4 * Z * (1 + 1.5 * Z))
    # ) / 2.0

    # expression_twostep_discontinuous_alpha1 = (
    #     1 + Z + np.sqrt((1 + Z) ** 2 - 4 * Z * Z)
    # ) / 2.0
    # expression_twostep_discontinuous_alpha2 = (
    #     1 + Z - np.sqrt((1 + Z) ** 2 - 4 * Z * Z)
    # ) / 2.0

    # 精确积分
    w = 0.6
    bb = -(1 + 5 / 3 * Z)
    cc = 2 / 3 * Z + 1 / 6 * Z * Z
    bb = -(1 + 1.5 * Z + (1 - w) / w + (0.5 * w - 1 / 3) * Z / w)
    cc = Z / w * (1 - 1.5 * (1 - w) + 2 * Z / 3 - 0.5 * w * (1 + 1.5 * Z)) + (
        1 + 1.5 * Z
    ) * ((1 - w) / w + (0.5 * w - 1 / 3) * Z / w)
    expression_twostep_discontinuous_alpha1 = (-bb + np.sqrt(bb * bb - 4 * cc)) / 2.0
    expression_twostep_discontinuous_alpha2 = (-bb - np.sqrt(bb * bb - 4 * cc)) / 2.0

    # 计算表达式的模

    stable_RK4, abs_rho_RK4 = stable_region_RK4(Z)
    stable_EulerForward, abs_rho_EulerForward = stable_region_EulerForward(Z)

    modulus_twostep_continuous_alpha1 = np.abs(expression_twostep_alpha1)
    modulus_twostep_continuous_alpha2 = np.abs(expression_twostep_alpha2)

    modulus_twostep_discontinuous_alpha1 = np.abs(
        expression_twostep_discontinuous_alpha1
    )
    modulus_twostep_discontinuous_alpha2 = np.abs(
        expression_twostep_discontinuous_alpha2
    )

    # 绘制图形
    plt.figure(figsize=(10, 10))

    plt.contour(X, Y, abs_rho_RK4, levels=[1], colors="red", linewidths=2)
    plt.contour(X, Y, abs_rho_EulerForward, levels=[1], colors="black", linewidths=2)

    val_twostep_continuous = 1.0
    mask_intersection = (modulus_twostep_continuous_alpha1 < val_twostep_continuous) & (
        modulus_twostep_continuous_alpha2 < val_twostep_continuous
    )
    plt.contourf(X, Y, mask_intersection, levels=[0.5, 1.5], colors="green", alpha=0.2)

    val_twostep_discontinuous = 1.0
    mask_intersection_testfunc = (
        modulus_twostep_discontinuous_alpha1 < val_twostep_discontinuous
    ) & (modulus_twostep_discontinuous_alpha2 < val_twostep_discontinuous)
    plt.contourf(
        X, Y, mask_intersection_testfunc, levels=[0.5, 1.5], colors="blue", alpha=0.2
    )

    plt.contour(
        X,
        Y,
        modulus_twostep_discontinuous_alpha1,
        levels=[val_twostep_discontinuous],
        colors="blue",
        linewidths=2,
    )
    plt.contour(
        X,
        Y,
        modulus_twostep_discontinuous_alpha2,
        levels=[val_twostep_discontinuous],
        colors="blue",
        linewidths=2,
    )

    # 添加标题和标签
    plt.xlabel("(Re(z))")
    plt.ylabel("(Im(z))")

    # 添加网格线以便更好地读取坐标
    plt.grid(True, linestyle="--", alpha=0.5)

    # 设置坐标轴范围
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # # 创建代理艺术家用于图例
    classical_rk4 = mlines.Line2D([], [], color="red", linewidth=2, label="RK4")
    euler = mlines.Line2D([], [], color="black", linewidth=2, label="Forward Euler")
    new_time_scheme_discontinuous = mlines.Line2D(
        [], [], color="blue", linewidth=2, label="new_time_scheme_discontinuous"
    )
    # 添加图例
    plt.legend(
        handles=[classical_rk4, euler, new_time_scheme_discontinuous],
        loc="best",
        fontsize=10,
    )

    plt.savefig("stable_region_newtimescheme.png")


if __name__ == "__main__":
    two_step_two_stage()
    cerk()
    newtimescheme()
    imex()
