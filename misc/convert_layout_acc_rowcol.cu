#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace cute;

int main()
{
	auto MMA_M = _8{};
	auto MMA_N = _2{};
	using L1 = decltype(
		make_layout(
			make_shape(_4{}, MMA_M, MMA_N),
			LayoutLeft{}
		)
	);

	using L2 = decltype(
		Shape<_2>{}
	);

	using L1_mode1 = decltype(
		Layout<
			Shape<_4>
		>{}
	);

	auto layout = logical_divide(L1{}, L2{});
	print(layout);
	printf("\n");

	auto layout_mode1 = logical_divide(L1_mode1{}, L2{});
	print(layout_mode1);
	printf("\n");

	return 0;
}