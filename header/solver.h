#pragma once
#include "config.h"
#include "element.h"
#include "macro.h"
#include "physics.h"
#include <memory>

class Solver
{
private:
    Geom *geom_pool;
    Element **elem_pool_old;
    Element **elem_pool_tmp;
    Rhs **rhs_pool_tmp;
    Config config;
    std::unique_ptr<PhysicsModel> physics;

public:
    Solver(const Config& config, int nelem);

    ~Solver();

    void Initialization();

    void computeRhs(Rhs *, const Element *);

    void computeElemRhsDG(Rhs *, const Element *, int);

    void computeElemRhsFR(Rhs *, const Element *, int);

    void compPredictionLP(const DataType (&flux)[NSP][NCONSRV],
                          const DataType &local_det_jac,
                          DataType (&rhs_predict)[NSP][NCONSRV]);

    void timeRK1();

    void timeNewExplicitSchemeK1();

    void timeRK2();

    void timeRK3();

    void TvdLimiter();

    std::pair<DataType, int> Minmod(DataType a, DataType b, DataType c);

    void Output(const std::string &);

    void OutputAvg(const std::string &);

    void computeElementGrad(int ielem);

    void ComputeElementAvg(int ielem);

    void compGradAndAvg();

    // 获取物理模型指针
    PhysicsModel* getPhysics() const { return physics.get(); }

    // 获取几何数据池
    Geom* getGeomPool() const { return geom_pool; }

    // 获取单元数据池
    Element** getElemPool() const { return elem_pool_old; }

    // 获取单元数量
    int getNumElems() const { return config.n_ele; }

    // 计算守恒变量在计算域中的体积分
    void computeConservationIntegral(DataType integral[NCONSRV]) const
    {
        physics->computeDomainIntegral(elem_pool_old[TORDER], geom_pool, config.n_ele, integral);
    }

    // 根据CFL数计算时间步长
    void computeCflDt();

    // 获取Config引用
    Config& getConfig() { return config; }
    const Config& getConfig() const { return config; }

    // 边界条件处理: 获取边界处的虚拟守恒变量和梯度
    // bc_pos: -1 表示左边界, +1 表示右边界
    void getBoundaryState(int bc_pos, DataType u_bc[NCONSRV], DataType u_grad_bc[NCONSRV]) const;
};
