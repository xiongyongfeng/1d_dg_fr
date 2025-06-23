#include "constants.h"
#include "element.h"
#include "parser.h"
#include "solver_config.h"
#include <cstdlib>
#include <iostream>
int main(int argc, char **argv) {
  std::cout << "running command: ";
  for (int i = 0; i < argc; i++) {
    std::cout << argv[i] << " ";
  }
  std::cout << std::endl;
  if (argc < 1) {
    std::cout << "usage: exe *.json" << std::endl;
    exit(-1);
  }

  Element ele;

  SolverConfig config = nlohmann::loadConfig(argv[1]);

  // 保存修改后的配置
  nlohmann::saveConfig(config, "config_updated.json");

  return 0;
}