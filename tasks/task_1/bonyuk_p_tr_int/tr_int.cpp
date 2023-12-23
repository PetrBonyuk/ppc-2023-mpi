#include "task_1/bonyuk_p_tr_int/tr_int.h"
#include <cmath>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

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
	const double step = std::abs(b - a) / N;
	const int steps_count = N / world.size();

	if (world.rank() == 0) {
		for (int proc = 1; proc < world.size(); proc++) {
			world.send(proc, 0, a + steps_count * step * proc);
		}
	}

	double local_val;
	if (world.rank() == 0) {
		local_val = a;
	}
	else {
		world.recv(0, 0, local_val);
	}
	double global_area = 0;
	double local_area = 0;

	if (world.rank() == world.size() - 1) {
		local_area = get_area(local_val, f, steps_count + N - steps_count * world.size(), step);
	}
	else {
		local_area = get_area(local_val, f, steps_count, step);
	}
	reduce(world, local_area, global_area, std::plus<double>(), 0);
	return global_area;
}