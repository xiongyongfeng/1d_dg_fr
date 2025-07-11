#pragma once
#include "config.h"
#include "element.h"
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
    void computeRhs(Rhs *, Element *);
    void computeElemRhs(Rhs *, Element *, int);
    void timeRK1();
    void timeRK2();
    void timeRK3();
    void Output(const std::string &);
    void computeElementGrad(int ielem);
};