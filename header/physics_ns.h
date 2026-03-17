#pragma once
#include "config.h"
#include "constants.h"
#include "physics.h"
#include <cmath>

/**
 * @brief 一维Euler方程 (Navier-Stokes 无粘项)
 * [rho, rho*u, rho*E]_t + [rho*u, rho*u^2+p, (rho*E+p)*u]_x = 0
 */
class PhysicsNS : public PhysicsModel<ConfigNS>
{
  public:
    // 单点通量计算
    void computeFlux(const DataType u[NCONSRV], DataType flux[NCONSRV],
                     const ConfigNS &config) const override
    {
        (void)config;
        DataType prim[NPRIMTV];
        cons2prim(u, prim);
        flux[0] = prim[0] * prim[1];
        flux[1] = prim[0] * prim[1] * prim[1] + prim[2];
        flux[2] = (u[2] + prim[2]) * prim[1];
    }

    void computeFluxNormal(const DataType u[NCONSRV], DataType flux[NCONSRV],
                           const ConfigNS &config,
                           DataType normal) const override
    {
        (void)config;
        DataType prim[NPRIMTV];
        cons2prim(u, prim);
        DataType un = prim[1] * normal;
        flux[0] = prim[0] * un;
        flux[1] = prim[0] * prim[1] * un + prim[2] * normal;
        flux[2] = (u[2] + prim[2]) * un;
    }

    void computeRiemannFluxLF(const DataType uL[NCONSRV],
                              const DataType uR[NCONSRV],
                              DataType flux[NCONSRV], const ConfigNS &config,
                              DataType normal) const
    {
        (void)config;
        DataType primtv_l[NPRIMTV];
        DataType primtv_r[NPRIMTV];
        cons2prim(uL, primtv_l);
        cons2prim(uR, primtv_r);
        DataType c_L = GetSoundSpeed(primtv_l);
        DataType c_R = GetSoundSpeed(primtv_r);
        DataType lamda_max =
            std::max(std::abs(primtv_l[1]) + c_L, std::abs(primtv_l[1]) + c_R);

        DataType flux_l[NCONSRV];
        DataType flux_r[NCONSRV];
        computeFluxNormal(uL, flux_l, config, normal);
        computeFluxNormal(uR, flux_r, config, normal);

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            flux[ivar] = DataType(0.5) * (flux_l[ivar] + flux_r[ivar]) -
                         DataType(0.5) * lamda_max * (uR[ivar] - uL[ivar]);
        }
    }

    void computeRiemannFluxHLL(const DataType uL[NCONSRV],
                               const DataType uR[NCONSRV],
                               DataType flux[NCONSRV], const ConfigNS &config,
                               DataType normal) const
    {
        (void)config;
        DataType primtv_l[NPRIMTV];
        DataType primtv_r[NPRIMTV];
        cons2prim(uL, primtv_l);
        cons2prim(uR, primtv_r);
        DataType c_L = GetSoundSpeed(primtv_l);
        DataType c_R = GetSoundSpeed(primtv_r);

        DataType un_l = primtv_l[1] * normal;
        DataType un_r = primtv_r[1] * normal;
        DataType flux_l[NCONSRV];
        DataType flux_r[NCONSRV];
        computeFluxNormal(uL, flux_l, config, normal);
        computeFluxNormal(uR, flux_r, config, normal);

        // HLL
        DataType lambda_L = std::min(un_l - c_L, un_r - c_R);
        DataType lambda_R = std::max(un_l + c_L, un_r + c_R);

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            if (lambda_L >= 0)
            {
                flux[ivar] = flux_l[ivar];
            }
            else if (lambda_R <= 0)
            {
                flux[ivar] = flux_r[ivar];
            }
            else
            {
                flux[ivar] =
                    (lambda_R * flux_l[ivar] - lambda_L * flux_r[ivar] +
                     lambda_L * lambda_R * (uR[ivar] - uL[ivar])) /
                    (lambda_R - lambda_L);
            }
        }
    }

    void computeRiemannFluxAUSM(const DataType consrv_l[NCONSRV],
                                const DataType consrv_r[NCONSRV],
                                DataType flux[NCONSRV], const ConfigNS &config,
                                DataType normal[SPACEDIM]) const
    {
        const double gamma = GAMMA;
        const double gm1 = gamma - 1.0;

        // --- 1. 恢复原始变量 (Left) ---
        double rhoL = consrv_l[0];
        double invRhoL = 1.0 / rhoL;
        double uL[3] = {0.0, 0.0, 0.0};
        double u2L = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            uL[i] = consrv_l[i + 1] * invRhoL;
            u2L += uL[i] * uL[i];
        }
        double eL = consrv_l[NCONSRV - 1] * invRhoL;
        double pL = gm1 * (eL - 0.5 * u2L) * rhoL;
        double aL = std::sqrt(gamma * pL * invRhoL);

        // --- 2. 恢复原始变量 (Right) ---
        double rhoR = consrv_r[0];
        double invRhoR = 1.0 / rhoR;
        double uR[3] = {0.0, 0.0, 0.0};
        double u2R = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            uR[i] = consrv_r[i + 1] * invRhoR;
            u2R += uR[i] * uR[i];
        }
        double eR = consrv_r[NCONSRV - 1] * invRhoR;
        double pR = gm1 * (eR - 0.5 * u2R) * rhoR;
        double aR = std::sqrt(gamma * pR * invRhoR);

        // --- 3. 法向速度与马赫数 ---
        double unL = 0.0, unR = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            unL += uL[i] * normal[i];
            unR += uR[i] * normal[i];
        }

        double ML = unL / aL;
        double MR = unR / aR;

        // --- 4. Splitting Functions ---
        auto getMPlus = [](double M)
        {
            if (std::abs(M) <= 1.0)
            {
                return 0.25 * (M + 1.0) * (M + 1.0) +
                       0.125 * (M * M - 1.0) * (M * M - 1.0);
            }
            return 0.5 * (M + std::abs(M));
        };

        auto getMMinus = [](double M)
        {
            if (std::abs(M) <= 1.0)
            {
                return -0.25 * (M - 1.0) * (M - 1.0) -
                       0.125 * (M * M - 1.0) * (M * M - 1.0);
            }
            return 0.5 * (M - std::abs(M));
        };

        auto getPPlus = [](double M)
        {
            if (std::abs(M) <= 1.0)
            {
                return 0.25 * (M + 1.0) * (M + 1.0) * (2.0 - M) +
                       0.1875 * M * (M * M - 1.0) * (M * M - 1.0);
            }
            return 0.5 * (1.0 + std::copysign(1.0, M));
        };

        auto getPMinus = [](double M)
        {
            if (std::abs(M) <= 1.0)
            {
                return 0.25 * (M - 1.0) * (M - 1.0) * (2.0 + M) -
                       0.1875 * M * (M * M - 1.0) * (M * M - 1.0);
            }
            return 0.5 * (1.0 - std::copysign(1.0, M));
        };

        double Mplus_L = getMPlus(ML);
        double Mminus_R = getMMinus(MR);
        double Pplus_L = getPPlus(ML);
        double Pminus_R = getPMinus(MR);

        // --- 5. Interface Quantities ---
        double M_half = Mplus_L + Mminus_R;
        double p_half = Pplus_L * pL + Pminus_R * pR;
        double a_half = std::min(aL, aR);

        double du_n = unR - unL;
        double p_diss =
            -0.25* Pplus_L * Pminus_R * (rhoL + rhoR) * a_half * du_n;
        double p_final = p_half + p_diss;

        // --- 6. Upwind State Selection ---
        const DataType *U = (M_half >= 0.0) ? consrv_l : consrv_r;
        double rho_up = U[0];
        double invRho_up = 1.0 / rho_up;

        double u2_up = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            double ui = U[i + 1] * invRho_up;
            u2_up += ui * ui;
        }
        double e_up = U[NCONSRV - 1] * invRho_up;
        double p_up = gm1 * (e_up - 0.5 * u2_up) * rho_up;
        double H_up = e_up + p_up * invRho_up; // Total Enthalpy
        double rhoH_up = rho_up * H_up;

        double Ma = M_half * a_half;

        // --- 7. Compute Flux ---
        flux[0] = Ma * rho_up; // Mass

        for (int i = 0; i < SPACEDIM; ++i)
        {
            flux[i + 1] = Ma * U[i + 1] + p_final * normal[i]; // Momentum
        }

        flux[NCONSRV - 1] = Ma * rhoH_up; // Energy
    }

    void computeRiemannFluxROE(const DataType consrv_l[NCONSRV],
                               const DataType consrv_r[NCONSRV],
                               DataType flux[NCONSRV], const ConfigNS &config,
                               DataType normal[SPACEDIM]) const
    {
        const DataType gamma = GAMMA;
        const DataType gm1 = gamma - 1.0;
        const DataType gpgm = gamma + 1.0; // gamma + 1
        const DataType gmgm = gamma - 1.0; // gamma - 1 (same as gm1)

        // --- 1. 提取左右状态的原始变量 ---
        // 守恒变量: [rho, rho*u, rho*v, rho*w, E]
        DataType rho_l = consrv_l[0];
        DataType rho_r = consrv_r[0];

        // 防止除以零或负密度
        if (rho_l <= 0.0 || rho_r <= 0.0)
        {
            // 处理真空或非物理状态，这里简单设为0或采用其他修复策略
            for (int i = 0; i < NCONSRV; ++i)
                flux[i] = 0.0;
            return;
        }

        DataType inv_rho_l = 1.0 / rho_l;
        DataType inv_rho_r = 1.0 / rho_r;

        // 速度
        DataType u_l[SPACEDIM], u_r[SPACEDIM];
        for (int i = 0; i < SPACEDIM; ++i)
        {
            u_l[i] = consrv_l[i + 1] * inv_rho_l;
            u_r[i] = consrv_r[i + 1] * inv_rho_r;
        }

        // 动能
        DataType ke_l = 0.5 * rho_l;
        DataType ke_r = 0.5 * rho_r;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            ke_l += 0.0; // 占位，下面统一算点积
            ke_r += 0.0;
        }
        // 重新计算动能: 0.5 * rho * |u|^2
        ke_l = 0.5 * rho_l;
        ke_r = 0.5 * rho_r;
        DataType u2_l = 0.0, u2_r = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            u2_l += u_l[i] * u_l[i];
            u2_r += u_r[i] * u_r[i];
        }
        ke_l *= u2_l;
        ke_r *= u2_r;

        // 压力 p = (gamma - 1) * (E - ke)
        DataType p_l = gm1 * (consrv_l[SPACEDIM + 1] - ke_l);
        DataType p_r = gm1 * (consrv_r[SPACEDIM + 1] - ke_r);

        // 总焓 H = (E + p) / rho
        DataType h_l = (consrv_l[SPACEDIM + 1] + p_l) * inv_rho_l;
        DataType h_r = (consrv_r[SPACEDIM + 1] + p_r) * inv_rho_r;

        // 法向速度 Un = u . n
        DataType un_l = 0.0, un_r = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            un_l += u_l[i] * normal[i];
            un_r += u_r[i] * normal[i];
        }

        // --- 2. 计算 Roe 平均量 ---
        // sqrt(rho)
        DataType sq_rho_l = std::sqrt(rho_l);
        DataType sq_rho_r = std::sqrt(rho_r);
        DataType sum_sq_rho = sq_rho_l + sq_rho_r;

        if (sum_sq_rho <= 1e-14)
        {
            for (int i = 0; i < NCONSRV; ++i)
                flux[i] = 0.0;
            return;
        }

        DataType inv_sum_sq_rho = 1.0 / sum_sq_rho;

        // Roe 平均速度
        DataType u_tilde[SPACEDIM];
        for (int i = 0; i < SPACEDIM; ++i)
        {
            u_tilde[i] =
                (sq_rho_l * u_l[i] + sq_rho_r * u_r[i]) * inv_sum_sq_rho;
        }

        // Roe 平均总焓
        DataType h_tilde = (sq_rho_l * h_l + sq_rho_r * h_r) * inv_sum_sq_rho;

        // Roe 平均法向速度
        DataType un_tilde = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
            un_tilde += u_tilde[i] * normal[i];

        // Roe 平均声速 c = sqrt((gamma-1)*(H - 0.5*|u|^2))
        DataType u2_tilde = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
            u2_tilde += u_tilde[i] * u_tilde[i];

        DataType c_tilde_sq = gm1 * (h_tilde - 0.5 * u2_tilde);
        if (c_tilde_sq < 0.0)
            c_tilde_sq = 0.0; // 数值保护
        DataType c_tilde = std::sqrt(c_tilde_sq);

        // --- 3. 计算物理通量 (F_L 和 F_R) ---
        // F = [rho*Un, rho*u*Un + p*n, (E+p)*Un]
        DataType flux_l[NCONSRV], flux_r[NCONSRV];

        // L state
        flux_l[0] = rho_l * un_l;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            flux_l[i + 1] = rho_l * u_l[i] * un_l + p_l * normal[i];
        }
        flux_l[SPACEDIM + 1] = (consrv_l[SPACEDIM + 1] + p_l) * un_l;

        // R state
        flux_r[0] = rho_r * un_r;
        for (int i = 0; i < SPACEDIM; ++i)
        {
            flux_r[i + 1] = rho_r * u_r[i] * un_r + p_r * normal[i];
        }
        flux_r[SPACEDIM + 1] = (consrv_r[SPACEDIM + 1] + p_r) * un_r;

        // --- 4. 特征波分解与耗散项 ---
        // Delta U = U_R - U_L
        DataType du[NCONSRV];
        for (int i = 0; i < NCONSRV; ++i)
            du[i] = consrv_r[i] - consrv_l[i];

        // 速度差在法向和切向的投影
        // du_vector = u_r - u_l
        DataType du_vec[SPACEDIM];
        for (int i = 0; i < SPACEDIM; ++i)
            du_vec[i] = u_r[i] - u_l[i];

        // 法向速度差
        DataType dun = 0.0;
        for (int i = 0; i < SPACEDIM; ++i)
            dun += du_vec[i] * normal[i];

        // 压力差
        DataType dp = p_r - p_l;

        // 密度差
        DataType drho = rho_r - rho_l;

        // 特征值 (波速)
        DataType lambda[3];
        lambda[0] = un_tilde - c_tilde;
        lambda[1] = un_tilde; // 多重根，对应熵波和剪切波
        lambda[2] = un_tilde + c_tilde;

        DataType rho_tilde =
            sq_rho_l * sq_rho_r; // geometric mean? No, usually arithmetic in
                                 // some forms, but here derived from H relation

        DataType rho_bar = sq_rho_l * sq_rho_r;

        // 为了避免复杂的特征向量重构，我们使用通量差分分裂的等效形式：
        // Dissipation = |A| * dU
        // 我们可以显式构建 |A|dU 的分量

        // 辅助量
        DataType abs_lambda1 = std::fabs(lambda[0]);
        DataType abs_lambda2 = std::fabs(lambda[1]);
        DataType abs_lambda3 = std::fabs(lambda[2]);

        // 为了防止除以零
        DataType c2 = c_tilde * c_tilde;
        if (c2 < 1e-12)
            c2 = 1e-12;

        DataType alpha1 = 0.5 * (dp - rho_bar * c_tilde * dun) / c2;
        DataType alpha3 = 0.5 * (dp + rho_bar * c_tilde * dun) / c2;
        DataType alpha2_density = drho - dp / c2;

        DataType diss[NCONSRV] = {0};

        // 计算切向速度差
        DataType du_t[SPACEDIM];
        for (int i = 0; i < SPACEDIM; ++i)
        {
            du_t[i] =
                du_vec[i] - dun * normal[i]; // Vector component tangential
        }

        // 开始组装耗散项 sum |lambda| * dw * r
        // 初始化
        for (int i = 0; i < NCONSRV; ++i)
            diss[i] = 0.0;

        // Wave 1 (u-c)
        DataType w1 = alpha1 * abs_lambda1;
        diss[0] += w1 * 1.0;
        for (int i = 0; i < SPACEDIM; ++i)
            diss[i + 1] += w1 * (u_tilde[i] - c_tilde * normal[i]);
        diss[SPACEDIM + 1] += w1 * (h_tilde - un_tilde * c_tilde);

        // Wave 3 (u+c)
        DataType w3 = alpha3 * abs_lambda3;
        diss[0] += w3 * 1.0;
        for (int i = 0; i < SPACEDIM; ++i)
            diss[i + 1] += w3 * (u_tilde[i] + c_tilde * normal[i]);
        diss[SPACEDIM + 1] += w3 * (h_tilde + un_tilde * c_tilde);

        DataType w2 = alpha2_density * abs_lambda2;
        diss[0] += w2 * 1.0;
        for (int i = 0; i < SPACEDIM; ++i)
            diss[i + 1] += w2 * u_tilde[i];
        diss[SPACEDIM + 1] += w2 * (0.5 * u2_tilde); // 0.5 * |u|^2

        // 剪切波 (Shear waves)
        // 只有在 SPACEDIM > 1 时存在
        if (SPACEDIM > 1)
        {

            // 计算切向速度差向量 du_t_vec (已在上面计算为 du_t)
            // 缩放因子
            DataType shear_scale = rho_bar * abs_lambda2;

            for (int i = 0; i < SPACEDIM; ++i)
            {
                // 动量方程：减去 rho * du_t_i
                // 注意方向：特征变量是 -du_t，所以贡献是 (-du_t) * r_shear.
                // r_shear 在动量分量上是 rho * (direction).
                // 所以总贡献是 - rho * du_t_i.
                diss[i + 1] += shear_scale * du_t[i];

                // 能量方程：u . (momentum_change)
                diss[SPACEDIM + 1] += shear_scale * (u_tilde[i] * du_t[i]);
            }
        }

        // --- 5. 最终通量 ---
        // F = 0.5 * (F_L + F_R) - 0.5 * Dissipation
        for (int i = 0; i < NCONSRV; ++i)
        {
            flux[i] = 0.5 * (flux_l[i] + flux_r[i]) - 0.5 * diss[i];
        }
    }

    void computeRiemannFluxHLLC(const DataType consrv_l[NCONSRV],
                                const DataType consrv_r[NCONSRV],
                                DataType flux[NCONSRV], const ConfigNS &config,
                                DataType normal[SPACEDIM]) const
    {
        const DataType gamma = static_cast<DataType>(GAMMA);
        const DataType gm1 = gamma - static_cast<DataType>(1.0);
        const DataType small_num = static_cast<DataType>(1.0e-10);

        // --- 1. 恢复原始变量 (Primitive Variables) ---

        // 左侧
        DataType primtv_l[NPRIMTV];
        cons2prim(consrv_l, primtv_l);
        DataType rho_l = primtv_l[0];
        if (rho_l < small_num)
            rho_l = small_num;

        DataType vel_l[SPACEDIM];
        DataType ke_l = static_cast<DataType>(0.0);
        for (int i = 0; i < SPACEDIM; ++i)
        {
            vel_l[i] = primtv_l[i + 1];
            ke_l += vel_l[i] * vel_l[i];
        }
        ke_l *= static_cast<DataType>(0.5) * rho_l;

        DataType E_l = consrv_l[SPACEDIM + 1];
        DataType p_l = primtv_l[SPACEDIM + 1];
        if (p_l < small_num)
            p_l = small_num;
        DataType a_l = std::sqrt(gamma * p_l / rho_l);

        // 右侧
        DataType primtv_r[NPRIMTV];
        cons2prim(consrv_r, primtv_r);
        DataType rho_r = primtv_r[0];
        if (rho_r < small_num)
            rho_r = small_num;

        DataType vel_r[SPACEDIM];
        DataType ke_r = static_cast<DataType>(0.0);
        for (int i = 0; i < SPACEDIM; ++i)
        {
            vel_r[i] = primtv_r[i + 1];
            ke_r += vel_r[i] * vel_r[i];
        }
        ke_r *= static_cast<DataType>(0.5) * rho_r;

        DataType E_r = consrv_r[SPACEDIM + 1];
        DataType p_r = primtv_r[SPACEDIM + 1];
        if (p_r < small_num)
            p_r = small_num;
        DataType a_r = std::sqrt(gamma * p_r / rho_r);

        // --- 2. 坐标旋转：计算法向速度和切向速度向量 ---

        // 法向速度 un = V . n
        DataType un_l = static_cast<DataType>(0.0);
        DataType un_r = static_cast<DataType>(0.0);
        for (int i = 0; i < SPACEDIM; ++i)
        {
            un_l += vel_l[i] * normal[i];
            un_r += vel_r[i] * normal[i];
        }

        // 切向速度向量 Vt = V - un * n
        // 我们直接存储 Vt 的分量，避免动态分配
        DataType vt_l[SPACEDIM];
        DataType vt_r[SPACEDIM];
        for (int i = 0; i < SPACEDIM; ++i)
        {
            vt_l[i] = vel_l[i] - un_l * normal[i];
            vt_r[i] = vel_r[i] - un_r * normal[i];
        }

        // --- 3. Roe 平均值计算波速 ---

        DataType sqrt_rho_l = std::sqrt(rho_l);
        DataType sqrt_rho_r = std::sqrt(rho_r);
        DataType inv_sum_sqrt_rho =
            static_cast<DataType>(1.0) / (sqrt_rho_l + sqrt_rho_r);

        DataType u_tilde =
            (sqrt_rho_l * un_l + sqrt_rho_r * un_r) * inv_sum_sqrt_rho;

        // 总焓 H = (E + p) / rho
        DataType H_l = (E_l + p_l) / rho_l;
        DataType H_r = (E_r + p_r) / rho_r;
        DataType H_tilde =
            (sqrt_rho_l * H_l + sqrt_rho_r * H_r) * inv_sum_sqrt_rho;

        DataType a_tilde_sq =
            gm1 * (H_tilde - static_cast<DataType>(0.5) * u_tilde * u_tilde);
        if (a_tilde_sq < static_cast<DataType>(0.0))
            a_tilde_sq = static_cast<DataType>(0.0);
        DataType a_tilde = std::sqrt(a_tilde_sq);

        // 波速 SL, SR
        DataType SL = std::min(un_l - a_l, u_tilde - a_tilde);
        DataType SR = std::max(un_r + a_r, u_tilde + a_tilde);

        // --- 4. 计算星区状态 (S*, p*) ---

        DataType rhoL_SL_uL = rho_l * (SL - un_l);
        DataType rhoR_SR_uR = rho_r * (SR - un_r);
        DataType denom = rhoL_SL_uL - rhoR_SR_uR;

        DataType S_star, p_star;

        if (std::abs(denom) < small_num)
        {
            // 退化情况处理
            S_star = static_cast<DataType>(0.5) * (un_l + un_r);
            p_star = static_cast<DataType>(0.5) * (p_l + p_r);
        }
        else
        {
            S_star =
                (p_r - p_l + rhoL_SL_uL * un_l - rhoR_SR_uR * un_r) / denom;
            p_star = p_l + rhoL_SL_uL * (S_star - un_l);
        }

        // 物理限制：防止负压
        if (p_star < static_cast<DataType>(0.0))
            p_star = static_cast<DataType>(0.0);

        // --- 5. 根据波区计算通量 ---

        DataType flux_rho, flux_mom_n, flux_E;
        DataType rho_star = static_cast<DataType>(0.0);
        int region = 0; // 0:Left, 1:StarL, 2:StarR, 3:Right

        if (SL >= static_cast<DataType>(0.0))
        {
            // --- 纯左状态 ---
            region = 0;
            flux_rho = rho_l * un_l;
            flux_mom_n = rho_l * un_l * un_l + p_l;
            flux_E = un_l * (E_l + p_l);
        }
        else if (S_star >= static_cast<DataType>(0.0))
        {
            // --- 左星区 ---
            region = 1;
            DataType div_term = SL - S_star;
            if (std::abs(div_term) < small_num)
                div_term = small_num; // 防除零

            rho_star = rhoL_SL_uL / div_term;

            flux_rho = rho_star * S_star;
            flux_mom_n = rho_star * S_star * S_star + p_star;

            // E* = p*/(gamma-1) + 0.5 * rho* * S*^2
            DataType E_star = p_star / gm1 + static_cast<DataType>(0.5) *
                                                 rho_star * S_star * S_star;
            flux_E = S_star * (E_star + p_star);
        }
        else if (SR >= static_cast<DataType>(0.0))
        {
            // --- 右星区 ---
            region = 2;
            DataType div_term = SR - S_star;
            if (std::abs(div_term) < small_num)
                div_term = small_num;

            rho_star = rhoR_SR_uR / div_term;

            flux_rho = rho_star * S_star;
            flux_mom_n = rho_star * S_star * S_star + p_star;

            DataType E_star = p_star / gm1 + static_cast<DataType>(0.5) *
                                                 rho_star * S_star * S_star;
            flux_E = S_star * (E_star + p_star);
        }
        else
        {
            // --- 纯右状态 ---
            region = 3;
            flux_rho = rho_r * un_r;
            flux_mom_n = rho_r * un_r * un_r + p_r;
            flux_E = un_r * (E_r + p_r);
        }

        // --- 6. 重构多维通量向量 ---

        // 选择正确的切向速度
        const DataType *vt_ptr = (region == 0 || region == 1) ? vt_l : vt_r;

        // 组装通量
        flux[0] = flux_rho; // 密度通量

        // 动量通量: F_mom = (法向动量通量)*n + (质量通量)*Vt
        for (int i = 0; i < SPACEDIM; ++i)
        {
            flux[i + 1] = flux_mom_n * normal[i] + flux_rho * vt_ptr[i];
        }

        flux[SPACEDIM + 1] = flux_E; // 能量通量
    }

    void computeRiemannFlux(const DataType uL[NCONSRV],
                            const DataType uR[NCONSRV], DataType flux[NCONSRV],
                            const ConfigNS &config,
                            DataType normal) const override
    {
        if (config.common_flux_type == CommonFluxType::HLL)
        {
            computeRiemannFluxHLL(uL, uR, flux, config, normal);
        }
        else if (config.common_flux_type == CommonFluxType::LF)
        {
            computeRiemannFluxLF(uL, uR, flux, config, normal);
        }
        else if (config.common_flux_type == CommonFluxType::HLLC)
        {
            DataType normal_1d[SPACEDIM];
            normal_1d[0] = normal;
            computeRiemannFluxHLLC(uL, uR, flux, config, normal_1d);
        }
        else if (config.common_flux_type == CommonFluxType::ROE)
        {
            DataType normal_1d[SPACEDIM];
            normal_1d[0] = normal;
            computeRiemannFluxROE(uL, uR, flux, config, normal_1d);
        }
        else if (config.common_flux_type == CommonFluxType::AUSM)
        {
            DataType normal_1d[SPACEDIM];
            normal_1d[0] = normal;
            computeRiemannFluxAUSM(uL, uR, flux, config, normal_1d);
        }
    }

    void prim2cons(const DataType prim[NPRIMTV],
                   DataType cons[NCONSRV]) const override
    {
        cons[0] = prim[0];
        cons[1] = prim[0] * prim[1];
        cons[2] = 0.5 * prim[0] * prim[1] * prim[1] + prim[2] / (GAMMA - 1);
    }

    void cons2prim(const DataType cons[NCONSRV],
                   DataType prim[NPRIMTV]) const override
    {
        prim[0] = cons[0];
        prim[1] = cons[1] / cons[0];
        prim[2] = (cons[2] - 0.5 * prim[0] * prim[1] * prim[1]) * (GAMMA - 1);
        prim[3] = prim[2] / (prim[0] * GAS_R);
    }

    void setInitialCondition(DataType u[NSP][NCONSRV], const DataType x[NSP],
                             const ConfigNS &config) const override
    {
        (void)config;
        DataType consrv_left[NCONSRV];
        DataType consrv_right[NCONSRV];
        DataType primtv_left[NPRIMTV];
        DataType primtv_right[NPRIMTV];

        // Sod problem initial condition
        primtv_left[0] = 1.0;
        primtv_left[1] = 0.0;
        primtv_left[2] = 1.0;
        primtv_left[3] = 1.0;

        primtv_right[0] = 0.125;
        primtv_right[1] = 0.0;
        primtv_right[2] = 0.1;
        primtv_right[3] = 0.1 / 0.125;

        prim2cons(primtv_left, consrv_left);
        prim2cons(primtv_right, consrv_right);

        for (int isp = 0; isp < NSP; isp++)
        {
            if (x[isp] < DataType(0.0))
            {
                u[isp][0] = consrv_left[0];
                u[isp][1] = consrv_left[1];
                u[isp][2] = consrv_left[2];
            }
            else
            {
                u[isp][0] = consrv_right[0];
                u[isp][1] = consrv_right[1];
                u[isp][2] = consrv_right[2];
            }
        }
    }

    std::string name() const override { return "NS"; }

    bool hasEntropyModify() const override { return true; }

    void compPredictionEntropy(const DataType flux[NSP][NCONSRV],
                               const DataType consrv[NSP][NCONSRV],
                               const DataType local_det_jac,
                               DataType rhs_predict[NSP][NCONSRV],
                               const ConfigNS &config) const override
    {
        (void)config;
        (void)flux;

        // Ismail & Roe entropy stable flux
        DataType primtv[NSP][NPRIMTV];
        for (int isp = 0; isp < NSP; isp++)
        {
            cons2prim(consrv[isp], primtv[isp]);
        }

        for (int isp = 0; isp < NSP; isp++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                DataType rho_R = primtv[isp][0];
                DataType rho_L = primtv[jsp][0];
                DataType p_R = primtv[isp][2];
                DataType p_L = primtv[jsp][2];
                DataType u_R = primtv[isp][1];
                DataType u_L = primtv[jsp][1];

                DataType z1_L = std::sqrt(rho_L / p_L);
                DataType z1_R = std::sqrt(rho_R / p_R);
                DataType z2_L = z1_L * u_L;
                DataType z2_R = z1_R * u_R;
                DataType z3_L = z1_L * p_L;
                DataType z3_R = z1_R * p_R;

                DataType z2_bar = 0.5 * (z2_L + z2_R);
                DataType z3_bar_log = z3_L;
                if (std::abs(z3_R - z3_L) / (std::abs(z3_L) + 1e-12) >= 1e-6)
                {
                    z3_bar_log =
                        (z3_R - z3_L) / (std::log(z3_R) - std::log(z3_L));
                }

                DataType fs_1 = z2_bar * z3_bar_log;
                rhs_predict[isp][0] -= 2.0 *
                                       getDMatrix<DataType, ORDER>()[isp][jsp] *
                                       fs_1 / local_det_jac;

                DataType z3_bar = 0.5 * (z3_L + z3_R);
                DataType z1_bar = 0.5 * (z1_L + z1_R);

                DataType fs_2 = z3_bar / z1_bar + z2_bar / z1_bar * fs_1;
                rhs_predict[isp][1] -= 2.0 *
                                       getDMatrix<DataType, ORDER>()[isp][jsp] *
                                       fs_2 / local_det_jac;

                DataType z1_bar_log = z1_L;
                if (std::abs(z1_R - z1_L) / (std::abs(z1_L) + 1e-12) >= 1e-6)
                {
                    z1_bar_log =
                        (z1_R - z1_L) / (std::log(z1_R) - std::log(z1_L));
                }
                DataType fs_3 =
                    0.5 * z2_bar / z1_bar *
                    ((GAMMA + 1.0) / (GAMMA - 1.0) * z3_bar_log / z1_bar_log +
                     fs_2);
                rhs_predict[isp][2] -= 2.0 *
                                       getDMatrix<DataType, ORDER>()[isp][jsp] *
                                       fs_3 / local_det_jac;
            }
        }
    }

    // 计算守恒变量在计算域中的体积分
    void computeDomainIntegral(const Element *elem_pool, const Geom *geom_pool,
                               int n_ele,
                               DataType integral[NCONSRV]) const override
    {
        // 初始化为0
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            integral[ivar] = DataType(0.0);
        }

        // 对每个单元求积分并累加
        // 使用LGL求积公式: ∫u dx = sum_i w_i * u_i * (dx/2)
        for (int iele = 0; iele < n_ele; iele++)
        {
            const Element &elem = elem_pool[iele];
            DataType local_det_jac = geom_pool[iele].local_det_jac;

            for (int isp = 0; isp < NSP; isp++)
            {
                DataType weight = getLGLWeights<DataType, ORDER>()[isp];
                for (int ivar = 0; ivar < NCONSRV; ivar++)
                {
                    integral[ivar] +=
                        weight * elem.u_consrv[isp][ivar] * local_det_jac;
                }
            }
        }
    }

  private:
    // Helper functions
    DataType GetSoundSpeed(const DataType prim[NPRIMTV]) const
    {
        return GAMMA * prim[2] / prim[0];
    }
};

// 工厂函数实现
inline std::unique_ptr<PhysicsModel<ConfigNS>> createPhysicsModelNS()
{
    return std::make_unique<PhysicsNS>();
}