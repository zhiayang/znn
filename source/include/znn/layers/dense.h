// dense.h
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
		// the dense layer only operates on the last dimension of the input tensor, leaving the
		// other dimensions intact. eg. if the input is 500x300x100, passing it through a Dense
		// would yield an output of 500x300xN
		template <size_t N, typename InputLayer, typename ActivationFn, typename RegulariserFn>
		struct Dense : Layer
		{
			Dense(InputLayer& input, ActivationFn af, RegulariserFn rf) : Layer(&input),
				activator(std::move(af)), regulariser(std::move(rf))
			{
				// fill with normally-distributed junk
				this->weights = xt::random::randn<double>(this->weights.shape(), 0, 1);
				this->transposedWeights = xt::transpose(this->weights);
			}

			using InputShape = typename InputLayer::OutputShape;
			using OutputShape = typename InputShape::template drop<1>::template add<N>;
			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");

			virtual xarr compute(bool training, bool batched) override
			{
				auto input = this->prev()->compute(training, batched);
				assert(ensure_correct_dimensions<InputShape>(input, batched));

				xarr output = util::matrix_mul(input, this->transposedWeights) + this->biases;
				assert(ensure_correct_dimensions<OutputShape>(output, batched));

				this->last_output = xt::eval(this->activator.forward(output));
				return this->last_output;
			}

			virtual void backward(const xarr& error, bool batched) override
			{
				assert(ensure_correct_dimensions<OutputShape>(error, batched));

				auto gradient = error * this->activator.derivative(this->last_output);
				auto newerror = util::matrix_mul(gradient, this->weights);

				if(batched)
				{
					// need to squash along the batch dimension
					this->update_dw_db(
						xt::sum(gradient, 0),
						xt::sum(this->prev()->getLastOutput(), 0)
					);
				}
				else
				{
					this->update_dw_db(gradient, this->prev()->getLastOutput());
				}

				this->prev()->backward(newerror, batched);
			}

			virtual void updateWeights(optimisers::Optimiser* opt, double scale) override
			{
				assert(zfu::equal(this->d_weight.shape(), weights.shape()));
				opt->computeDeltas(this, this->d_weight, this->d_bias);

				// also regularise the weight, which applies a penalty for large weights
				// to combat overfitting.
				this->weights -= scale * (this->d_weight + this->regulariser.derivative(this->weights));

				// note that we need to collapse the incoming db; each bias contributes
				// to multiple nodes' errors downstream, so we just do the simple thing and
				// sum them up.
				this->biases -= scale * xt::sum(this->d_bias, 0);

				this->transposedWeights = xt::transpose(this->weights);

				this->prev()->updateWeights(opt, scale);
			}

		private:
			ActivationFn activator;
			RegulariserFn regulariser;
			xt::xtensor_fixed<double, xt::xshape<N>> biases;
			xt::xtensor_fixed<double, xt::xshape<N, InputShape::template last<>>> weights;
			xt::xtensor_fixed<double, xt::xshape<InputShape::template last<>, N>> transposedWeights;
		};
	}

	// the order of templates like this is so that InputLayer never needs to be specified
	// and we can specify the rest of the templates, eg. activation.
	template <size_t N, typename AF = activations::Linear, typename RF = regularisers::None, typename InputLayer>
	impl::Dense<N, InputLayer, AF, RF> Dense(InputLayer& il, const AF& af = AF(), const RF& rf = RF())
	{
		return impl::Dense<N, InputLayer, AF, RF>(il, af, rf);
	}
}
