#include <cute/tensor.hpp>
using namespace cute;

int main()
{
	using LayoutAtom = decltype(Layout<
								Shape<_4, _8>,
								Stride<_8, _1>>{});

	using Layout = decltype(tile_to_shape(
		LayoutAtom{},
		Shape<_8, _16>{}));

	using LayoutT = decltype(composition(
		Layout{},
		make_layout(
			Shape<_16, _8>{}, GenRowMajor{})));

	using LayoutSwizzleAtom = decltype(composition(
		Swizzle<3, 0, 3>{},
		LayoutAtom{}));

	using LayoutSwizzle = decltype(tile_to_shape(
		LayoutSwizzleAtom{},
		Shape<_8, _16>{}));

	using LayoutSwizzleT = decltype(composition(
		LayoutSwizzle{},
		make_layout(
			Shape<_16, _8>{}, GenRowMajor{})));

	using LayoutSwizzleTNoSwizzle = decltype(get_nonswizzle_portion(LayoutSwizzleT{}));

	print_layout(Layout{});
	print_layout(LayoutSwizzle{});
	print_layout(LayoutT{});
	print_layout(LayoutSwizzleT{});
	print_layout(LayoutSwizzleTNoSwizzle{});

	return 0;
}