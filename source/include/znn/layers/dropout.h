// dropout.h
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
		struct Dropout : Layer
		{
			Dropout(InputLayer& input, double probability) : Layer(&input), probability(probability)
			{
				assert(0 <= probability && probability < 1.0);
			}

			using InputShape = typename InputLayer::OutputShape;
			using OutputShape = InputShape;

			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");


			virtual xarr compute(bool training, bool batched) override
			{
				auto input = this->prev()->compute(training, batched);
				assert(ensure_correct_dimensions<InputShape>(input, batched));

				if(training)
				{
					auto perform = [&](auto& m) -> xarr {
						// we need to scale it by 1/(1-P) to keep the expected sum of the output values
						// the same regardless of the dropout probability
						m /= (1.0 - this->probability);
						return m * input;
					};

					// first generate the mask. we lose nodes with P probability, so we want to generate
					// a binomial distribution with 1-P chance of success.
					auto&& rands = xt::random::binomial<double>(input.shape(), 1, 1.0 - this->probability);
					if(batched)
					{
						this->batchedMask = std::move(rands);
						this->last_output = perform(this->batchedMask);
					}
					else
					{
						this->mask = std::move(rands);
						this->last_output = perform(this->mask);
					}
				}
				else
				{
					this->last_output = std::move(input);
				}

				return this->last_output;
			}

			virtual void backward(const xarr& error, bool batched) override
			{
				assert(ensure_correct_dimensions<OutputShape>(error, batched));

				// since we have no weights, there's no need to update dw or db.
				if(batched) this->prev()->backward(this->batchedMask * error, batched);
				else        this->prev()->backward(this->mask * error, batched);
			}

			virtual void updateWeights(optimisers::Optimiser* opt, double scale) override
			{
				// just passthrough
				this->prev()->updateWeights(opt, scale);
			}

		private:
			double probability = 0;

			// xtensor_fixed demands an xshape shape, which demands variadic template args; we can't
			// convert our znn::shape to that, so the best we can do is fix the number of dimensions.
			xt::xtensor<double, InputShape::dims> mask;
			xt::xtensor<double, 1 + InputShape::dims> batchedMask;
		};
	}

	template <typename InputLayer>
	impl::Dropout<InputLayer> Dropout(InputLayer& il, double prob)
	{
		return impl::Dropout<InputLayer>(il, prob);
	}
}
