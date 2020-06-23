// batchnorm.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "../misc.h"
#include "../activations.h"
#include "../regularisers.h"

namespace znn::layers
{
	namespace impl
	{
		template <typename InputLayer>
		struct BatchNorm : Layer
		{
			BatchNorm(InputLayer& input, double momentum = 0.999, double epsilon = 1e-8)
				: Layer(&input), momentum(momentum), epsilon(epsilon)
			{
				assert(epsilon > 0);
				assert(0 < momentum && momentum <= 1);
			}

			using InputShape = typename InputLayer::OutputShape;
			using OutputShape = InputShape;

			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");




			virtual xarr compute(bool training) override
			{
				assert(this->prev());
				auto input = this->prev()->compute(training);
				assert(zfu::equal(input.shape(), InputShape::sizes));


				return this->last_output;
			}

			virtual std::pair<xarr, xarr> backward(const xarr& delta) override
			{
				assert(zfu::equal(delta.shape(), this->last_output.shape()));

				return { };
			}

			virtual void updateWeights(const xarr& dw, const xarr& db, double scale) override
			{
			}

		private:
			const double momentum = 0;
			const double epsilon = 0;
		};
	}


}
