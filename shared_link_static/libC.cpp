#include "libE.h"

extern void callPrintMessageFromD();

void callPrintMessageFromC() {
    printMessage();
    callPrintMessageFromD();
}