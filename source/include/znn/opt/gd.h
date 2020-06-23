// gd.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "../cost.h"
#include "../misc.h"
#include "../model.h"

/*
	implementation of "vanilla" mini-batch gradient descent. you usually never want
	to use this, it just serves as a generic base for more advanced optimisers. at
	the very least you should use StochasticGD, which is this but with momentum.

	for now, the Specialisation only gets to customise the weight updating. fortunately
	that's more than enough to implement sgd, rmsprop and adam. this precludes us from
	implementing nesterov momentum (since that affects the initial backprop), but that's
	fine.
*/

namespace znn::optimisers
{
	template <typename CostFn, typename Specialisation>
	struct GDDriver
	{
		GDDriver(size_t batchSize, double learningRate, Specialisation& spec)
			: batchSize(batchSize), learningRate(learningRate), spec(spec)
		{
			rng = std::mt19937(util::getSeed());
		}

	protected:
		const size_t batchSize = 0;
		const double learningRate = 0;
		Specialisation& spec;

		std::mt19937 rng;

		struct layer_deltas_t
		{
			xarr d_weight;
			xarr d_bias;
		};

		using LayerDeltaMap = std::unordered_map<Layer*, layer_deltas_t>;

		void train_one_sample(Model& model, LayerDeltaMap& deltas, const xarr& input, const xarr& target)
		{
			auto prediction = model.train_forward(input);
			auto out_layer = model.outputLayer();

			std::unordered_map<Layer*, xarr> gradients;
			{
				xarr delta = this->spec.costFn.derivative(target, prediction);

				auto cl = out_layer;
				while(cl && cl->prev())
				{
					auto [ gradient, new_delta ] = cl->backward(delta);
					// gradients[cl] = std::move(gradient);
					delta = std::move(new_delta);

					deltas[cl].d_weight += util::dot(gradient, cl->prev()->getLastOutput());
					deltas[cl].d_bias   += gradient;

					cl = cl->prev();
				}
			}

			// {
			// 	auto cl = out_layer;
			// 	while(cl && cl->prev()) // don't do the input layer
			// 	{
			// 		auto w = util::dot(gradients[cl], cl->prev()->getLastOutput());

			// 		auto& dels = deltas[cl];
			// 		dels.d_weight += std::move(w);
			// 		dels.d_bias   += std::move(gradients[cl]);

			// 		cl = cl->prev();
			// 	}
			// }
		}

	public:
		void run(Model& model, const std::vector<xarr>& inputs, const std::vector<xarr>& targets)
		{
			assert(inputs.size() == targets.size());

			size_t count = inputs.size();
			size_t index = 0;
			auto indices = std::vector<size_t>(count);
			{
				std::iota(indices.begin(), indices.end(), 0);
				std::shuffle(indices.begin(), indices.end(), this->rng);
			}

			size_t remaining = count;
			size_t todo = std::min(remaining, this->batchSize);

			// let the specialisation setup any per-batch metrics (eg. velocity)
			this->spec.setup();

			while(todo > 0)
			{
				LayerDeltaMap deltas;

				for(size_t i = 0; i < todo; i++)
				{
					this->train_one_sample(model, deltas, inputs[indices[index]], targets[indices[index]]);
					index++;
				}

				this->spec.update_weights(deltas, todo, model.outputLayer());

				remaining -= todo;
				todo = std::min(remaining, this->batchSize);
			}
		}
	};


	template <typename CostFn>
	struct VanillaGD : GDDriver<CostFn, VanillaGD<CostFn>>
	{
		using Base = GDDriver<CostFn, VanillaGD<CostFn>>;
		using LayerDeltaMap = typename Base::LayerDeltaMap;

		friend Base;

		VanillaGD(size_t batchSize, double learningRate, CostFn costfn = CostFn())
			: Base(batchSize, learningRate, *this), costFn(std::move(costfn)) { }


	private:
		CostFn costFn;

		void setup()
		{
			// do nothing
		}

		void update_weights(LayerDeltaMap& deltas, size_t samples, Layer* cl)
		{
			while(cl && cl->prev())
			{
				auto& [ dw, db ] = deltas[cl];

				cl->updateWeights(dw, db, 1.0 / ((double) samples / this->learningRate));
				cl = cl->prev();
			}
		}
	};
}
