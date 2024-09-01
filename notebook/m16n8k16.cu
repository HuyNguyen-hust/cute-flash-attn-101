#include <cute/tensor.hpp>

using namespace cute;

int main()
{
	using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
	print_latex(MMA_Atom_Arch{});
	return 0;
}