// main.cpp
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#include <stdio.h>

#include "znn/znn.h"


#if 1

int main(int argc, char** argv)
{
	using namespace znn;

	optimisers::ENABLE_BATCHED() = (argc > 1 && std::string(argv[1]) == "batch");

	znn::util::setSeed(1);
	xt::print_options::set_line_width(90);
	xt::print_options::set_precision(9);

	auto in = layers::Input<shape<2>>();
	auto a = layers::Dense<10>(in, activations::Sigmoid());
	auto b = layers::BatchNorm(a);
	// auto c = layers::Flatten(a);
	auto d = layers::Dense<1, activations::Sigmoid>(b);
	auto model = Model(in, d);

	std::vector<xarr> inputs;
	std::vector<xarr> outputs;

	inputs.reserve(4 * 500);
	outputs.reserve(4 * 500);



	for(size_t i = 0; i < 200; i++)
	{
		inputs.push_back({ 0, 0 }); outputs.push_back({ 0 });
		inputs.push_back({ 0, 1 }); outputs.push_back({ 1 });
		inputs.push_back({ 1, 0 }); outputs.push_back({ 1 });
		inputs.push_back({ 1, 1 }); outputs.push_back({ 0 });
	}

	auto opt = optimisers::Adam<cost::MeanSquare>(4, 0.01);
	for(size_t i = 0; i < 50; i++)
	{
		fprintf(stderr, "\r            \repoch %zu", i + 1);
		// fprintf(stderr, "\n");

		znn::train(model, inputs, outputs, opt);
	}

	fprintf(stderr, "\n\n\n");

	std::cout << "0 ^ 0  =  " << xt::flatten(model.predict({ 0, 0 })) << "\n";
	std::cout << "0 ^ 1  =  " << xt::flatten(model.predict({ 0, 1 })) << "\n";
	std::cout << "1 ^ 0  =  " << xt::flatten(model.predict({ 1, 0 })) << "\n";
	std::cout << "1 ^ 1  =  " << xt::flatten(model.predict({ 1, 1 })) << "\n";

	std::cout << "\n";

	// std::cout << d.weights << "\n";

	printf("hello, world!\n");
}
#else

int main()
{
	using namespace znn;
	auto a = xt::arange(15).reshape({ 3, 5 });
	auto b = xt::arange(84).reshape({ 4, 3, 7 });

	// std::cout << xt::adapt(xt::linalg::tensordot(xt::transpose(a), b,
	// 	{ a.dimension() - 1 }, { 1 }).shape()) << "\n";

	std::cout << util::matrix_mul(xt::transpose(a), b) << "\n";
}

#endif
