#pragma once
#include "config.h"
#include "element.h"
class Solver
{
  private:
    Element *elem_pool;
    Config config;

  public:
    Solver(const Config &config, Element *elem_pool)
    {
        this->elem_pool = elem_pool;
        this->config = config;
    };
    ~Solver(){};

    void Initialization();
    void computeRhs();
    void timeRK1();
    void timeRK3();
    void Output(const std::string &);
    void computeElementGrad(int ielem);
};