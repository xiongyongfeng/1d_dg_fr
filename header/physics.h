#pragma once
#include "macro.h"
#include "config.h"
#include "element.h"
#include <string>
#include <memory>

/**
 * @brief 物理模型抽象基类
 *
 * 所有方程（LAD、Burgers、NS等）都需要实现以下接口：
 * - 通量计算
 * - Riemann通量计算
 * - 原始变量与守恒变量转换
 * - 初始条件设置
 */
class PhysicsModel
{
public:
    virtual ~PhysicsModel() = default;

    // 通量计算: F = f(u) - 单点输入
    virtual void computeFlux(const DataType u[NCONSRV],
                            DataType flux[NCONSRV],
                            const Config& config) const = 0;

    // Riemann通量计算 (界面通量)
    virtual void computeRiemannFlux(const DataType uL[NCONSRV],
                                    const DataType uR[NCONSRV],
                                    DataType flux[NCONSRV],
                                    const Config& config) const = 0;

    // 原始变量 -> 守恒变量
    virtual void prim2cons(const DataType prim[NPRIMTV],
                          DataType cons[NCONSRV]) const = 0;

    // 守恒变量 -> 原始变量
    virtual void cons2prim(const DataType cons[NCONSRV],
                          DataType prim[NPRIMTV]) const = 0;

    // 设置初始条件 - 多点
    virtual void setInitialCondition(DataType u[NSP][NCONSRV],
                                     const DataType x[NSP],
                                     const Config& config) const = 0;

    // 模型名称
    virtual std::string name() const = 0;

    // 是否有扩散项（粘性）
    virtual bool hasDiffusion() const { return false; }

    // 粘性通量计算 (可选)
    virtual void computeVisFlux(const DataType u[NCONSRV],
                                const DataType u_grad[NCONSRV],
                                DataType flux[NCONSRV],
                                const Config& config) const
    {
        (void)u; (void)u_grad; (void)flux; (void)config;
    }

    // 粘性通量黎曼求解器 (可选) - 用于处理粘性界面的数值通量
    virtual void computeVisRiemannFlux(const DataType uL[NCONSRV],
                                       const DataType uL_grad[NCONSRV],
                                       const DataType uR[NCONSRV],
                                       const DataType uR_grad[NCONSRV],
                                       const DataType &length_scale,
                                       DataType flux[NCONSRV],
                                       const Config& config) const
    {
        (void)uL; (void)uL_grad; (void)uR; (void)uR_grad; (void)flux; (void)config;
    }

    // BR2 粘性通量计算 (可选)
    virtual void computeBR2Flux(const DataType uL[NCONSRV],
                                const DataType uL_grad[NCONSRV],
                                const DataType uR[NCONSRV],
                                const DataType uR_grad[NCONSRV],
                                DataType local_det_jac_L,
                                DataType local_det_jac_R,
                                DataType flux[NCONSRV],
                                DataType globalLift_L[NSP * NCONSRV],
                                DataType globalLift_R[NSP * NCONSRV],
                                const Config& config) const
    {
        (void)uL; (void)uL_grad; (void)uR; (void)uR_grad;
        (void)local_det_jac_L; (void)local_det_jac_R;
        (void)flux; (void)globalLift_L; (void)globalLift_R; (void)config;
    }

    // 熵修正预测 (可选)
    virtual void compPredictionEntropy(const DataType flux[NSP][NCONSRV],
                                      const DataType consrv[NSP][NCONSRV],
                                      const DataType local_det_jac,
                                      DataType rhs_predict[NSP][NCONSRV],
                                      const Config& config) const
    {
        (void)flux; (void)consrv; (void)local_det_jac;
        (void)rhs_predict; (void)config;
    }

    // 是否支持熵修正
    virtual bool hasEntropyModify() const { return false; }

    // 计算守恒变量在计算域中的体积分
    // @param elem_pool: 单元数据池
    // @param geom_pool: 几何数据池
    // @param n_ele: 单元数量
    // @return: 每个守恒变量的积分值数组
    virtual void computeDomainIntegral(const Element* elem_pool,
                                      const Geom* geom_pool,
                                      int n_ele,
                                      DataType integral[NCONSRV]) const
    {
        (void)elem_pool; (void)geom_pool; (void)n_ele;
        for (int ivar = 0; ivar < NCONSRV; ivar++)
            integral[ivar] = DataType(0.0);
    }
};

// 工厂函数：根据宏定义创建物理模型
std::unique_ptr<PhysicsModel> createPhysicsModel();
