// znn.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "precompile.h"

#include "misc.h"
#include "layer.h"
#include "activation.h"

namespace znn
{
	struct Model
	{
		Model(Layer& input_layer, Layer& output_layer) : input_layer(input_layer), output_layer(output_layer) { }

		xarr predict(const xarr& in)
		{
			auto* layer = &this->input_layer;
			xarr current = in;

			while(layer)
			{
				current = layer->forward(current);
				layer = layer->output_layer;
			}

			return current;
		}


		xarr back_propagate(const xarr& input, const xarr& target)
		{
			std::vector<xarr> saved;
			// saved.push_back(input);

			xarr current = input;

			auto* cl = &this->input_layer;
			while(cl)
			{
				current = cl->forward(current);
				saved.push_back(current);

				cl = cl->output_layer;
			}

			auto prediction = current;
			assert(target.shape() == prediction.shape());

			// root mean square error
			xarr error = xt::square(target - prediction);
			error = xt::sum(error) / error.shape()[0];
			error = xt::sqrt(error);


			std::vector<xarr> gradients;
			gradients.resize(saved.size());

			{
				xarr delta = target - prediction;
				size_t i = saved.size() - 1;

				auto* cl = &this->output_layer;
				while(cl)
				{
					auto [ gradient, new_delta ] = cl->backward(saved[i], delta);
					delta = new_delta;

					gradients[i] = gradient;

					cl = cl->input_layer;
					i -= 1;
				}
			}

			// zpr::println("%zu/%zu saved", saved.size(), gradients.size());
			// for(auto& g : gradients)
			// 	std::cout << xt::adapt(g.shape()) << "\n";

			{
				auto rate = 0.5;

				size_t i = 0;
				auto* cl = this->input_layer.output_layer;
				while(cl)
				{
					auto delta = rate * saved[i];
					auto w = util::dot(gradients[i+1], delta);

					cl->update_weights(w, delta);

					cl = cl->output_layer;
					i += 1;
				}
			}

			return error;






			// auto ret = error;
			// {
			// 	std::vector<xarr> deltas;
			// 	deltas.resize(saved.size());

			// 	auto* cl = &this->output_layer;
			// 	size_t i = saved.size() - 1;

			// 	while(cl)
			// 	{
			// 		auto [ delta, prev_error ] = cl->backward(saved[i], error);
			// 		deltas[i] = delta;

			// 		// std::cout << "perror: " << prev_error << "\n";
			// 		// std::cout << "delta: " << delta << "\n\n";

			// 		// update the error for the preceeding node
			// 		error = prev_error;

			// 		cl = cl->input_layer;
			// 		i -= 1;
			// 	}

			// 	{
			// 		auto rate = 0.5;

			// 		size_t i = 0;
			// 		auto cl = &this->input_layer;

			// 		while(cl)
			// 		{
			// 			auto w = deltas[i] * saved[i] * rate;
			// 			cl->update_weights(w, deltas[i] * rate);

			// 			cl = cl->output_layer;
			// 			i += 1;
			// 		}
			// 	}
			// }

			// return ret;
			return { };
		}





	#if 0
		xarr back_propagate(const xarr& input, const xarr& output)
		{
			std::vector<xarr> saved;
			saved.push_back(input);

			xarr current = input;

			auto* cl = &this->input_layer;
			while(cl)
			{
				current = cl->forward(current);
				saved.push_back(current);

				cl = cl->output_layer;
			}

			auto prediction = current;
			assert(output.shape() == prediction.shape());

			xarr error = (prediction - output);
			auto ret = error;
			{
				std::vector<xarr> deltas;
				deltas.resize(saved.size());

				auto* cl = &this->output_layer;
				size_t i = saved.size() - 1;

				while(cl)
				{
					auto [ delta, prev_error ] = cl->backward(saved[i], error);
					deltas[i] = delta;

					// std::cout << "perror: " << prev_error << "\n";
					// std::cout << "delta: " << delta << "\n\n";

					// update the error for the preceeding node
					error = prev_error;

					cl = cl->input_layer;
					i -= 1;
				}

				{
					auto rate = 0.5;

					size_t i = 0;
					auto cl = &this->input_layer;

					while(cl)
					{
						auto w = deltas[i] * saved[i] * rate;
						cl->update_weights(w, deltas[i] * rate);

						cl = cl->output_layer;
						i += 1;
					}
				}
			}

			return ret;
		}
	#endif

		Layer& input_layer;
		Layer& output_layer;
	};
}
