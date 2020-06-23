// main.cpp
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#include <stdio.h>

#include "znn/znn.h"


/*
	TODO list


	1. backpropagation should be done the same way as forward propagation

		for forward we can just do output_layer->compute(), and that will make
		sure that all its input layers get computed, and so forth. in declaring
		the layers we already made a dependency graph, so might as well use it.

		once models get complex enough that they're no longer a line, but a graph,
		it's impossible to do (current = current->prev()) cos there might be more
		than one previous layer.

		it might not be that hard to do this, since the gradient accumulation
		and weight/bias calculations are both done in the same direction, and are
		calculated the same way independent of the optimiser.


		also, worthwhile effort to do the backprop and the calc of dw and db in
		the same pass, so (a) we don't need to store gradients in a temporary, and
		(b) we can see how to structure the code to put it into the layer's backward
		pass. ofc we'll need a mechanism to generalise it the layers don't end up
		having copies of the same code.


	2. training needs to be done across all samples in a batch, layer by layer.

		currently, training is done in batches, but we do a full forward+backward
		pass on each model per sample. instead, what we need to do is to feed the
		entire batch of inputs to the first layer of all models, followed by running
		the second layer of all models, etc.

		not entirely sure how this can be done, since each layer has a bunch of
		internal state. the other issue is that since we are going with the "pull"
		approach instead of the "push" approach, it won't be that easy to do the
		naive thing and just loop through all models and save each layer's output
		into a temporary to feed into the next layer's input in the next loop.

		this approach is required for us to implement batch normalisation, which is
		apparently a Big Thing (tm). computing the activations of a batchnorm layer
		require the outputs from the previous layer *across the entire batch*, which
		definitely means that we must compute the entire batch layerwise rather than
		samplewise.



	3. add more layers

		probably next is conv1d and conv2d, after we finish batchnorm and batchrenorm.
		then some pooling layers (maxpool, minpool, avgpool) and i think that should
		be the MVP for this library.
*/













int main(int argc, char** argv)
{
	using namespace znn;

	znn::util::setSeed(1);

	auto in = layers::Input<shape<2>>();
	auto a = layers::Dense<10>(in, activations::Sigmoid(), regularisers::L2(0.001));
	auto b = layers::Dropout(a, 0.05);
	auto c = layers::Dense<1, activations::Sigmoid>(b);
	auto model = Model(in, c);

	std::vector<xarr> inputs;
	std::vector<xarr> outputs;

	inputs.reserve(4 * 500);
	outputs.reserve(4 * 500);

	for(size_t i = 0; i < 50; i++)
	{
		inputs.push_back({ 0, 0 }); outputs.push_back({ 0 });
		inputs.push_back({ 0, 1 }); outputs.push_back({ 1 });
		inputs.push_back({ 1, 0 }); outputs.push_back({ 1 });
		inputs.push_back({ 1, 1 }); outputs.push_back({ 0 });
	}

	auto opt = optimisers::Adam<cost::MeanSquare>(8, 0.1, 0.9);
	for(size_t i = 0; i < 50; i++)
	{
		fprintf(stderr, "\r            \repoch %zu", i);
		znn::train(model, inputs, outputs, opt);
	}

	fprintf(stderr, "\n");

	std::cout << "0 ^ 0  =  " << model.predict({ 0, 0 }) << "\n";
	std::cout << "0 ^ 1  =  " << model.predict({ 0, 1 }) << "\n";
	std::cout << "1 ^ 0  =  " << model.predict({ 1, 0 }) << "\n";
	std::cout << "1 ^ 1  =  " << model.predict({ 1, 1 }) << "\n";

	std::cout << "\n";




	printf("hello, world!\n");
}
