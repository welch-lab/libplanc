//
// Created by andrew on 3/14/2024.
//

#include "utils.h"

std::map<std::string, algotype> algomap{{"MU", MU},
                                        {"HALS", HALS},
                                        {"ANLSBPP", ANLSBPP},
                                        {"NAIVEANLSBPP", NAIVEANLSBPP},
                                        {"AOADMM", AOADMM},
                                        {"NESTEROV", NESTEROV},
                                        {"CPALS", CPALS},
                                        {"GNSYM", GNSYM},
                                        {"R2", R2},
                                        {"PGD", PGD},
                                        {"PGNCG", PGNCG}};
std::map<std::string, normtype> normmap{{"NONE", NONE}, {"L2NORM", L2NORM}, {"MAXNORM", MAXNORM}};