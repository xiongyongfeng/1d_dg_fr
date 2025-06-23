#pragma once
#include "constants.h"
#include "macro.h"

class Element {
public:
  Element(){};
  ~Element(){};

private:
  DataType u_consrv[NSP];
  DataType u_grad_consrv[NSP];
  DataType x[NSP];
};