// adam.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "gd.h"

namespace znn::optimisers
{
	template <typename CostFn>
	struct Adam : GDDriver<CostFn, Adam<CostFn>>
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
		using LayerDeltaMap = typename Base::LayerDeltaMap;

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
		}

		void update_weights(LayerDeltaMap& deltas, size_t samples, Layer* cl)
		{
			timestep += 1.0;

			while(cl && cl->prev())
			{
				auto& [ dw, db ] = deltas[cl];
				auto& [ g1, g2 ] = params[cl];

				g1 = (this->beta1 * g1) + ((1.0 - this->beta1) * dw);
				g2 = (this->beta2 * g2) + ((1.0 - this->beta2) * xt::square(dw));

				xarr mk = g1 / (1.0 - std::pow(beta1, timestep));
				xarr rk = g2 / (1.0 - std::pow(beta2, timestep));

				dw = mk / (xt::sqrt(rk) + this->epsilon);

				cl->updateWeights(dw, db, 1.0 / ((double) samples / this->learningRate));
				cl = cl->prev();
			}
		}
	};
}




