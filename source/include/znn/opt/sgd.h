// sgd.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "gd.h"

namespace znn::optimisers
{
	template <typename CostFn>
	struct StochasticGD : GDDriver<CostFn, StochasticGD<CostFn>>, Optimiser
	{
		using Base = GDDriver<CostFn, StochasticGD<CostFn>>;
		friend Base;

		StochasticGD(size_t batchSize, double learningRate, double momentum = 0.9, CostFn costfn = CostFn())
			: Base(batchSize, learningRate, *this), costFn(std::move(costfn)), momentum(momentum)
		{
		}

	private:
		using LayerVelocityMap = std::unordered_map<Layer*, xarr>;

		CostFn costFn;
		const double momentum = 0;
		LayerVelocityMap velocities;

		void setup()
		{
		}


		virtual void computeDeltas(Layer* layer, xarr& dw, xarr& db) override
		{
			if(this->momentum > 0)
			{
				xarr& vel = this->velocities[layer];
				vel = (this->momentum * vel) + dw;

				dw = vel;
				(void) db;
			}
		}

		void update_weights(size_t samples, Layer* last)
		{
			last->updateWeights(this, this->learningRate / (double) samples);
			last->resetDeltas();
		}
	};
}
