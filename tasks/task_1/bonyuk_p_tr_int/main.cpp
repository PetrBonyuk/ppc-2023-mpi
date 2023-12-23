#include "gtest/gtest.h"
#include <iostream>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include "task_1/bonyuk_p_tr_int/tr_int.h"

#define ERRORRATE 0.0001

TEST(TrapecIntegral, Fconst) {
	boost::mpi::communicator world;
	const int N = 1000;
	const double a = 0, b = 10;
	const double real_var = 10;

	double global_sum = TrapecIntegr(a, b, const_function, N);

	if (world.rank() == 0) {
		ASSERT_NEAR(global_sum, real_var, ERRORRATE);
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