// flatten.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "base.h"

#include "../activations.h"
#include "../regularisers.h"

namespace znn::layers
{
	namespace impl
	{
		template <typename InputLayer>
		struct Flatten : Layer
		{
			Flatten(InputLayer& input) : Layer(&input)
			{
			}

			using InputShape = typename InputLayer::OutputShape;
			using OutputShape = shape<InputShape::flatten()>;

			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");

			virtual xarr compute(bool training) override
			{
				assert(this->prev());
				auto input = this->prev()->compute(training);
				assert(zfu::equal(input.shape(), InputShape::sizes));

				this->last_output = xt::flatten(input);
				return this->last_output;
			}

			virtual void backward(const xarr& error) override
			{
				assert(zfu::equal(error.shape(), this->last_output.shape()));

				xarr gradient = error;
				gradient.reshape(InputShape::sizes);

				xarr newerror = gradient;

				// there's no need to call update_dw_db here, since we have no weights nor biases
				this->prev()->backward(newerror);

				// this->d_weight += util::matrix_mul(xt::transpose(gradient), this->prev()->getLastOutput());
				// this->d_bias += gradient;

				// return { gradient, newerror };
			}

			virtual void updateWeights(optimisers::Optimiser* opt, double scale) override
			{
				// there's nothing to do
				(void) opt;
				(void) scale;
			}

		private:
		};
	}

	template <typename InputLayer>
	impl::Flatten<InputLayer> Flatten(InputLayer& il)
	{
		return impl::Flatten<InputLayer>(il);
	}
}
