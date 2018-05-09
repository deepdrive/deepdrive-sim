#include "DeepDriveUtils.h"
#include "DeepDrive.h"


std::random_device DeepdriveUtilsRandomSeed;
std::mt19937 DeepdriveUtilsRandomGenerator(DeepdriveUtilsRandomSeed());    // random-number engine used (Mersenne-Twister in this case)
