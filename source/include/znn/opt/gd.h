// gd.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "../cost.h"
#include "../util.h"
#include "../model.h"

// the definition of the interface Optimiser lives in there, for reasons.
#include "../layers/base.h"

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

		void train_one_sample(Model& model, const xarr& input, const xarr& target)
		{
			model.feed_training(input);
			auto out_layer = model.outputLayer();
			auto prediction = out_layer->compute(/* training: */ true, /* batched: */ false);

			assert(target.shape() == prediction.shape());

			xarr error = this->spec.costFn.derivative(target, prediction);
			out_layer->backward(error, /* batched: */ false);
		}

	public:
		void run(Model& model, const std::vector<xarr>& inputs, const std::vector<xarr>& targets)
		{
			assert(inputs.size() == targets.size());
			if(inputs.empty())
				return;

			auto batched_shape = [this](auto& shape) -> std::vector<size_t> {
				auto ret = std::vector<size_t>(shape.begin(), shape.end());
				ret.insert(ret.begin(), this->batchSize);
				return ret;
			};

			auto x_shape = batched_shape(inputs[0].shape());
			auto y_shape = batched_shape(targets[0].shape());

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
				// update the batch dimension to be correct
				x_shape[0] = todo;
				y_shape[0] = todo;

				xarr x_batch = xarr::from_shape(x_shape);
				xarr y_batch = xarr::from_shape(y_shape);

				for(size_t i = 0; i < todo; i++)
				{
					xt::view(x_batch, i) = inputs[indices[index]];
					xt::view(y_batch, i) = targets[indices[index]];
					index++;
				}

				{
					model.feed_training(x_batch);
					auto out_layer = model.outputLayer();
					auto prediction = out_layer->compute(/* training: */ true, /* batched: */ true);

					assert(y_batch.shape() == prediction.shape());

					xarr error = this->spec.costFn.derivative(y_batch, prediction);
					out_layer->backward(error, /* batched: */ true);
				}

				this->spec.update_weights(todo, model.outputLayer());

				remaining -= todo;
				todo = std::min(remaining, this->batchSize);
			}
		}
	};

	template <typename CostFn>
	struct VanillaGD : GDDriver<CostFn, VanillaGD<CostFn>>, Optimiser
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

		virtual void computeDeltas(Layer* layer, xarr& dw, xarr& db) override
		{
			// vanilla gradient descent doesn't need to do anything special.
			(void) layer;
			(void) dw;
			(void) db;
		}

		void update_weights(size_t samples, Layer* last)
		{
			last->updateWeights(this, this->learningRate / (double) samples);
			last->resetDeltas();
		}
	};
}
