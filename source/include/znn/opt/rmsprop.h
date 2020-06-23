// rmsprop.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "gd.h"

namespace znn::optimisers
{
	template <typename CostFn>
	struct RMSProp : GDDriver<CostFn, RMSProp<CostFn>>, Optimiser
	{
		using Base = GDDriver<CostFn, RMSProp<CostFn>>;
		friend Base;

		RMSProp(size_t batchSize, double learningRate, double decay = 0.9, double epsilon = 1e-8, CostFn costfn = CostFn())
			: Base(batchSize, learningRate, *this), costFn(std::move(costfn)), decay(decay), epsilon(epsilon)
		{
		}

	private:
		using LayerGradMap = std::unordered_map<Layer*, xarr>;
		using LayerDeltaMap = typename Base::LayerDeltaMap;

		CostFn costFn;
		const double decay = 0;
		const double epsilon = 0;
		LayerGradMap history;

		void setup()
		{
		}

		virtual void computeDeltas(Layer* layer, xarr& dw, xarr& db) override
		{
			xarr& hist = this->history[layer];
			hist = (this->decay * hist) + ((1 - this->decay) * xt::square(dw));

			dw /= (xt::sqrt(hist) + this->epsilon);
			(void) db;
		}

		void update_weights(LayerDeltaMap& deltas, size_t samples, Layer* last)
		{
			last->updateWeights(this, 1.0 / ((double) samples / this->learningRate));
			last->resetDeltas();

			// while(cl && cl->prev())
			// {
			// 	auto& [ dw, db ] = deltas[cl];

			// 	{
			// 		xarr& hist = this->history[cl];
			// 		hist = (this->decay * hist) + ((1 - this->decay) * xt::square(dw));

			// 		dw /= (xt::sqrt(hist) + this->epsilon);
			// 	}

			// 	cl->updateWeights(dw, db, 1.0 / ((double) samples / this->learningRate));
			// 	cl = cl->prev();
			// }
		}
	};
}




