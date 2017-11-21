#pragma once
#include <random>

extern std::random_device DeepdriveUtilsRandomSeed;
extern std::mt19937 DeepdriveUtilsRandomGenerator;

static double RandomDouble(double start, double end)
{
	std::uniform_real_distribution<double> uni(start, end); // guaranteed unbiased
	auto random_double = uni(DeepdriveUtilsRandomGenerator);
	return random_double;
}
