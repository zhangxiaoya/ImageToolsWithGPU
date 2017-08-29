#ifndef __GLOBAL_CONSISTANT_CONFIGURE__
#define __GLOBAL_CONSISTANT_CONFIGURE__
#include <cstdint>
#include "LogPrinter/LogPrinter.hpp"

const unsigned WIDTH = 320;
const unsigned HEIGHT = 256;
const unsigned BYTESIZE = 2;
const unsigned WHOLESIZE = WIDTH * HEIGHT;

LogPrinter logPrinter;

uint8_t* allImageDataOnHost[WHOLESIZE];
uint8_t* allImageDataOnDevice[WHOLESIZE];
#endif
