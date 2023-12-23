#include <cmath>
#include <functional>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include "task_1/bonyuk_p_tr_int/tr_int.h"

double const_function(double x) {
	return 1;
}

double standard_function(double x) {
	return 1 / (1 + x * x);
}

double complex_function(double x) {
	return 3 / (1 + std::pow(x * x + 2 * x + 5, 2));
}

double complex_sqrt_function(double x) {
	if (x + 2 < 0) return std::numeric_limits<double>::quiet_NaN();
	return (4 + std::pow(2 * x + 6, 3)) / std::sqrt(x + 2);
}

double complex_sin_cos_function(double x) {
	return (std::pow(cos(x), 3) + 1) / (1 + std::pow(sin(x), 2));
}

double trapezium(double a, double b, functional f) {
	return (f(a) + f(b))*std::abs(b - a) / 2;
}

double get_area(double a, functional f, int steps_count, double step) {
	double res = 0;
	for (int i = 0; i < steps_count; i++) {
		res += trapezium(a + i * step, a + (i + 1) * step, f);
	}
	return res;
}

double TrapecIntegr(double a, double b, functional f, int N) {
	boost::mpi::communicator world;
	const double step = (b - a) / N;
	const int steps_per_proc = N / world.size();
	const int leftover_steps = N % world.size();

	double local_a = a + world.rank() * steps_per_proc * step;
	double local_b = local_a + steps_per_proc * step;

	if (world.rank() == world.size() - 1) {
		local_b += leftover_steps * step;
	}

	double local_area = get_area(local_a, f, steps_per_proc + (world.rank() == world.size() - 1 ? leftover_steps : 0), step);
	double global_area = 0;

	reduce(world, local_area, global_area, std::plus<double>(), 0);
	return global_area;
}