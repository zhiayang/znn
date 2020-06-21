// main.cpp
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#include <stdio.h>

#include "znn/znn.h"


int main(int argc, char** argv)
{
	using namespace znn;

	xt::random::seed(1);

	auto in = layers::Input<shape<2>>();
	auto a = layers::Dense<5, activations::Sigmoid>(in);
	auto b = layers::Dense<1, activations::Sigmoid>(a);
	auto model = Model(in, b);

	for(size_t i = 0; i < 10000; i++)
	{
		auto e1 = model.back_propagate({ 0, 0 }, { 0 });
		auto e2 = model.back_propagate({ 0, 1 }, { 1 });
		auto e3 = model.back_propagate({ 1, 0 }, { 1 });
		auto e4 = model.back_propagate({ 1, 1 }, { 0 });

		std::cout << i << ": " << e1(0) << ", " << e2(0) << ", " << e3(0) << ", " << e4(0) << "\n";
	}

	fprintf(stderr, "\n");

	std::cout << "0 ^ 0  =  " << model.predict({ 0, 0 }) << "\n";
	std::cout << "0 ^ 1  =  " << model.predict({ 0, 1 }) << "\n";
	std::cout << "1 ^ 0  =  " << model.predict({ 1, 0 }) << "\n";
	std::cout << "1 ^ 1  =  " << model.predict({ 1, 1 }) << "\n";

	std::cout << "\n";

	std::cout << a.weights << "\n\n";
	std::cout << b.weights << "\n\n";



	// x.biases = { 0.18, 0.10, 0.21, 0.97 };
	// x.weights = {
	// 	{ 0.54, 0.00, 0.13 },
	// 	{ 0.27, 0.12, 0.57 },
	// 	{ 0.42, 0.67, 0.89 },
	// 	{ 0.84, 0.82, 0.20 },
	// };









	printf("hello, world!\n");
}
