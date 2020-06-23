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

			virtual xarr compute(bool training, bool batched) override
			{
				auto input = this->prev()->compute(training, batched);
				assert(ensure_correct_dimensions<InputShape>(input, batched));

				this->last_output = (batched
					? xarr(xt::reshape_view(input, { input.shape()[0], OutputShape::sizes[0] }))
					: xarr(xt::flatten(input))
				);

				return this->last_output;
			}

			virtual void backward(const xarr& error, bool batched) override
			{
				assert(ensure_correct_dimensions<OutputShape>(error, batched));

				xarr newerror;
				if(batched)
				{
					std::vector<size_t> shape;
					shape.reserve(error.shape().size() + 1);
					shape.push_back(error.shape()[0]);
					shape.insert(shape.end(), InputShape::sizes.begin(), InputShape::sizes.end());

					newerror = xt::reshape_view(error, shape);
				}
				else
				{
					newerror = xt::reshape_view(error, InputShape::sizes);
				}


				// there's no need to call update_dw_db here, since we have no weights nor biases
				this->prev()->backward(newerror, batched);
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
