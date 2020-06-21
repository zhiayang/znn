// layer.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "misc.h"
#include "activation.h"

namespace znn::layers
{
	namespace impl
	{
		template <typename InputShape>
		struct Input : Layer
		{
			Input() : Layer(nullptr) { }

			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");
			using OutputShape = InputShape;


			virtual std::pair<xarr, xarr> backward(const xarr&, const xarr&) override
			{
				// do nothing, we have no weights.
				return { };
			}

			virtual xarr forward(const xarr& input) override
			{
				return input;
			}

			virtual void update_weights(const xarr&w, const xarr& del) override
			{
			}
		};

		// the dense layer only operates on the last dimension of the input tensor, leaving the
		// other dimensions intact. eg. if the input is 500x300x100, passing it through a Dense
		// would yield an output of 500x300xN
		template <size_t N, typename InputLayer, typename ActivationFn>
		struct Dense : Layer
		{
			Dense(InputLayer& input) : Layer(&input)
			{
				input.output_layer = this;

				// fill with normally-distributed junk
				this->weights = xt::random::randn<double>(this->weights.shape(), 0, 1);
			}

			using InputShape = typename InputLayer::OutputShape;
			using OutputShape = typename InputShape::template drop<1>::template add<N>;
			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");

			virtual xarr forward(const xarr& input) override
			{
				auto& is = input.shape();
				assert(std::equal(is.begin(), is.end(), InputShape::sizes.begin(), InputShape::sizes.end()));

				auto output = xarr::from_shape(OutputShape::sizes);
				for(size_t i = 0; i < N; i++)
				{
					auto ov = xt::view(output, xt::all(), i);
					auto wv = xt::view(weights, i, xt::all());

					assert(wv.shape() == input.shape());
					ov = xt::eval(biases(i) + xt::sum(wv * input));
				}

				return ActivationFn::forward(output);
			}

			// should return { gradient, new_delta }
			virtual std::pair<xarr, xarr> backward(const xarr& last, const xarr& delta) override
			{
				// std::cout << xt::adapt(delta.shape()) << " != " << xt::adapt(last.shape()) << "\n";
				assert(delta.shape() == last.shape());

				auto gradient = delta * ActivationFn::derivative(last);
				auto newdelta = util::dot(xt::transpose(this->weights), gradient);

				return { gradient, newdelta };
			}

			virtual void update_weights(const xarr& w, const xarr& del) override
			{
				// std::cout << xt::adapt(w.shape()) << " => " << xt::adapt(xarr(weights).shape()) << "\n";
				assert(w.shape() == weights.shape());

				// this does a broadcast.
				this->weights += w;
				// this->biases += del;
			}

		// private:
			xt::xtensor_fixed<double, xt::xshape<N>> biases;
			xt::xtensor_fixed<double, xt::xshape<N, InputShape::template last<>>> weights;
		};
	}


	template <typename InputShape>
	impl::Input<InputShape> Input()
	{
		return impl::Input<InputShape>();
	}

	template <size_t N, typename AF = activations::None, typename InputLayer>
	impl::Dense<N, InputLayer, AF> Dense(InputLayer& il)
	{
		return impl::Dense<N, InputLayer, AF>(il);
	}
}
