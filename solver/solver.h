#pragma once
#include "../base/element.h"
#include "../base/solver_config.h"
class Solver
{
  private:
    Element *elem_pool;
    SolverConfig solver_config;

  public:
    Solver(const SolverConfig &solver_config, Element *elem_pool)
    {
        this->elem_pool = elem_pool;
        this->solver_config = solver_config;
    };
    ~Solver(){};

    void Initialization();
    void Compute();
    void Output(const std::string &);
};