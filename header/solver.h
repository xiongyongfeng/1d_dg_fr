#pragma once
#include "config.h"
#include "element.h"
#include "macro.h"
class Solver
{
  private:
    Geom *geom_pool;
    Element *elem_pool_old;
    Element *elem_pool_tmp;
    Rhs *rhs_pool_tmp;
    Config config;

  public:
    Solver(const Config &config, int nelem)
    {
        this->elem_pool_old = new Element[nelem];
        this->elem_pool_tmp = new Element[nelem];
        this->config = config;
        this->rhs_pool_tmp = new Rhs[nelem];
        this->geom_pool = new Geom[nelem];
    };
    ~Solver()
    {
        delete[] elem_pool_old;
        delete[] elem_pool_tmp;
        delete[] rhs_pool_tmp;
        delete[] geom_pool;
    };

    void Initialization();
    void computeRhs(Rhs *, const Element *);
    void computeElemRhsDG(Rhs *, const Element *, int);
    void computeElemRhsFR(Rhs *, const Element *, int);
    void compPredictionLP(const DataType (&flux)[NSP][NCONSRV],
                          const DataType &local_det_jac,
                          DataType (&rhs_predict)[NSP][NCONSRV]);
    void compPredictionEntropy(const DataType (&flux)[NSP][NCONSRV],
                               const DataType (&consrv)[NSP][NCONSRV],
                               const DataType &local_det_jac,
                               DataType (&rhs_predict)[NSP][NCONSRV]);

    void compPredictionCR(const DataType (&consrv)[NSP][NCONSRV],
                          const DataType (&consrv_grad)[NSP][NCONSRV],
                          DataType (&rhs_predict)[NSP][NCONSRV]);
    void timeRK1();
    void timeRK2();
    void timeRK3();
    void TvdLimiter();
    std::pair<DataType, int> Minmod(DataType a, DataType b, DataType c);
    void Output(const std::string &);
    void OutputAvg(const std::string &);
    void computeElementGrad(int ielem);
    void ComputeElementAvg(int ielem);
    void compGradAndAvg();
};