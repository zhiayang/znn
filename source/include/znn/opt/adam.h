// adam.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "gd.h"

namespace znn::optimisers
{
	template <typename CostFn>
	struct Adam : GDDriver<CostFn, Adam<CostFn>>, Optimiser
	{
		using Base = GDDriver<CostFn, Adam<CostFn>>;
		friend Base;

		Adam(size_t batchSize, double learningRate,
			double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, CostFn costfn = CostFn())
				: Base(batchSize, learningRate, *this)
				, costFn(std::move(costfn)), beta1(beta1), beta2(beta2), epsilon(epsilon)
		{
		}

	private:
		using LayerGradMap = std::unordered_map<Layer*, xarr>;

		CostFn costFn;
		const double beta1 = 0;
		const double beta2 = 0;
		const double epsilon = 0;

		struct Params
		{
			xarr grad_avg;
			xarr grad2_avg;
		};

		double timestep = 0;
		std::unordered_map<Layer*, Params> params;

		void setup()
		{
			// i have no idea if we're supposed to reset this every epoch or not...
			// it's worse if we reset it, so let's just not reset it for now
			// this->timestep = 0;
		}

		virtual void computeDeltas(Layer* layer, xarr& dw, xarr& db) override
		{
			auto& [ g1, g2 ] = this->params[layer];

			g1 = (this->beta1 * g1) + ((1.0 - this->beta1) * dw);
			g2 = (this->beta2 * g2) + ((1.0 - this->beta2) * xt::square(dw));

			xarr mk = g1 / (1.0 - std::pow(beta1, timestep));
			xarr rk = g2 / (1.0 - std::pow(beta2, timestep));

			dw = mk / (xt::sqrt(rk) + this->epsilon);
			(void) db;
		}

		void update_weights(size_t samples, Layer* last)
		{
			timestep += 1.0;
			last->updateWeights(this, this->learningRate / (double) samples);
			last->resetDeltas();
		}
	};
}




