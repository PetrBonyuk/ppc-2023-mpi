﻿#include "gtest/gtest.h"
#include <iostream>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include "task_1/bonyuk_p_tr_int/tr_int.h"

#define ERRORRATE 0.1

TEST(TrapecIntegral, Fconst) {
	boost::mpi::communicator world;
	const int N = 1000;
	const double a = 0, b = 10;
	const double real_var = 10;

	double global_sum = TrapecIntegr(a, b, const_function, N);

	if (rank == 0) {
		ASSERT_LT(std::abs(real_var - global_sum), ERRORRATE);
	}
}

TEST(TrapecIntegral, standardfun) {
	boost::mpi::communicator world;
	const int N = 1000;
	const double a = -1;
	const double b = 1;
	const double real_var = 1.5708;


	double global_sum = TrapecIntegr(a, b, standard_function, N);

	if (rank == 0) {
		ASSERT_LT(std::abs(real_var - global_sum), ERRORRATE);
	}
}

TEST(TrapecIntegral, complfun) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	const int N = 1000;
	const double a = -1;
	const double b = 1;
	const double real_var = 0.230422;

	double global_sum = TrapecIntegr(a, b, complex_function, N);

	if (rank == 0) {
		ASSERT_LT(std::abs(real_var - global_sum), ERRORRATE);
	}
}

TEST(TrapecIntegral, sqrtfun) {
	boost::mpi::communicator world;
	const int N = 1000;
	const double a = -1;
	const double b = 1;
	const double real_var = 329.363;

	double global_sum = TrapecIntegr(a, b, complex_sqrt_function, N);

	if (rank == 0) {
		ASSERT_LT(std::abs(real_var - global_sum), ERRORRATE);
	}
}

TEST(TrapecIntegral, sincosfun) {
	boost::mpi::communicator world;
	const int N = 1000;
	const double a = -3.141592653589793;
	const double b = 3.141592653589793;
	const double real_var = 4.44288;

	double global_sum = TrapecIntegr(a, b, complex_sin_cos_function, N);

	if (rank == 0) {
		ASSERT_LT(std::abs(real_var - global_sum), ERRORRATE);
	}
}

int main(int argc, char** argv) {
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
	::testing::InitGoogleTest(&argc, argv);
	::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
	if (world.rank() != 0) {
		delete listeners.Release(listeners.default_result_printer());
	}
	return RUN_ALL_TESTS();
}
