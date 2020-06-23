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
					// first generate the mask. we lose nodes with P probability, so we want to generate
					// a binomial distribution with 1-P chance of success.
					// also note: the mask should be strictly the size of the input, unbatched.
					this->mask = xt::random::binomial(unbatched_input_shape(input, batched),
						1, 1.0 - this->probability);

					// then we need to scale it by 1/(1-P) to keep the expected sum of the output values
					// the same regardless of the dropout probability
					this->mask /= (1.0 - this->probability);

					// then mask the output.
					this->last_output = xt::eval(input * this->mask);
				}
				else
				{
					this->last_output = input;
				}

				return this->last_output;
			}

			virtual void backward(const xarr& error, bool batched) override
			{
				assert(ensure_correct_dimensions<OutputShape>(error, batched));

				// since we have no weights, there's no need to update dw or db.
				this->prev()->backward(error * this->mask, batched);
			}

			virtual void updateWeights(optimisers::Optimiser* opt, double scale) override
			{
				// there's nothing to do
				(void) opt;
				(void) scale;
			}

		private:
			double probability = 0;

			// xtensor_fixed demands an xshape shape, which demands variadic template args; we can't
			// convert our znn::shape to that, so the best we can do is fix the number of dimensions.
			xt::xtensor<double, InputShape::dims> mask;
		};
	}

	template <typename InputLayer>
	impl::Dropout<InputLayer> Dropout(InputLayer& il, double prob)
	{
		return impl::Dropout<InputLayer>(il, prob);
	}
}
