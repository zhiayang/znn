// dense.h
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
			}

			using InputShape = typename InputLayer::OutputShape;
			using OutputShape = typename InputShape::template drop<1>::template add<N>;
			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");

			virtual xarr compute(bool training) override
			{
				assert(this->prev());
				auto input = this->prev()->compute(training);

				assert(zfu::equal(input.shape(), InputShape::sizes));

				auto output = xarr::from_shape(OutputShape::sizes);
				for(size_t i = 0; i < N; i++)
				{
					auto ov = xt::view(output, xt::all(), i);
					auto wv = xt::view(weights, i, xt::all());

					assert(zfu::equal(wv.shape(), input.shape()));
					ov = xt::eval(biases(i) + xt::sum(wv * input));
				}

				this->last_output = xt::eval(this->activator.forward(output));
				return this->last_output;
			}

			virtual std::pair<xarr, xarr> backward(const xarr& delta) override
			{
				assert(zfu::equal(delta.shape(), this->last_output.shape()));

				auto gradient = delta * this->activator.derivative(this->last_output);
				auto newdelta = util::dot(xt::transpose(this->weights), gradient);

				return { gradient, newdelta };
			}

			virtual void updateWeights(const xarr& dw, const xarr& db, double scale) override
			{
				assert(zfu::equal(dw.shape(), weights.shape()));

				// also regularise the weight, which applies a penalty for large weights
				// to combat overfitting.
				this->weights -= scale * (dw + this->regulariser.derivative(this->weights));
				this->biases -= scale * db;
			}

		private:
			ActivationFn activator;
			RegulariserFn regulariser;
			xt::xtensor_fixed<double, xt::xshape<N>> biases;
			xt::xtensor_fixed<double, xt::xshape<N, InputShape::template last<>>> weights;
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
